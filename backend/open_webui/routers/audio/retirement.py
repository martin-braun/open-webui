"""Optional bridge policy: retire microphone items only after acknowledged text coverage.

The caller serializes command/observe/pump with the upstream writer. No method waits
for provider acknowledgements; the receive loop and microphone must remain live.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from uuid import uuid4

log = logging.getLogger(__name__)
SYNC_TIMEOUT = 15
DEPENDENCY_TIMEOUT = 30


def safe_id(value):
    return re.sub(r'[^a-zA-Z0-9_.:-]', '?', str(value))[:128]


def text_digest(text):
    return hashlib.sha256(text.strip().encode()).digest()


@dataclass
class InputAudio:
    order: int
    digest: bytes | None = None
    rejected: bool = False
    covered: bool = False
    established: bool = False
    message_id: str | None = None
    assigned: bool = False
    superseded: bool = False


class InputAudioRetirement:
    def __init__(self, protocol, call_id, retention_count):
        if type(retention_count) is not int or not 1 <= retention_count <= 4096:
            raise ValueError('Invalid input audio retention count')
        self.protocol = protocol
        self.call_id = call_id
        self.retention_count = retention_count
        self.selected_order = None
        self.unsupported_audio_parts = False
        self.inputs = {}
        # Tombstones prevent late events from resurrecting recordings. Bounded by the call limit.
        self.retired = set()
        self.responses = {}
        self.call_inputs = {}
        self.continuations = set()
        self.pending = None
        self.sending = None
        self.context = None
        self.context_id = None
        self.operation = None
        self.remove_context = None
        self.blocked = False
        self.wait_since = None
        self.last_dependencies = None
        self.cancelled_requests = []

    def input(self, item_id):
        if not isinstance(item_id, str) or not 0 < len(item_id) <= 256:
            raise ValueError('Invalid microphone item identity')
        if item_id in self.retired:
            return None
        if item_id not in self.inputs:
            if len(self.inputs) + len(self.retired) >= 4096:
                raise ValueError('Call limit reached. Start a new call.')
            self.inputs[item_id] = InputAudio(len(self.inputs) + len(self.retired))
        return self.inputs[item_id]

    def native_input(self, item_id):
        item = self.inputs.get(item_id)
        return item_id if item and not item.superseded and not item.rejected else None

    def recent_window(self, anchor):
        # Only selection advances the window. A future commit, or late ASR for an
        # older commit, must not move the acoustic context of an already selected turn.
        if anchor is None:
            return set()
        successful = sorted(
            (item.order, item_id)
            for item_id, item in self.inputs.items()
            if item.digest and not item.rejected and item.order <= anchor
        )
        return {item_id for _, item_id in successful[-self.retention_count :]}

    def command(self, event):
        """Return passthrough provider commands; hold context and generation at the barrier."""
        kind = event.get('type')
        if kind == 'bridge.context':
            if set(event) != {'type', 'messages', 'coverage', 'call_id'} or event['call_id'] != self.call_id:
                raise ValueError('Invalid call history identity')
            # Reuse the existing role/size allowlist and snapshot text, with private metadata stripped.
            commands = self.protocol.command({'type': kind, 'messages': event['messages']})
            coverage = event['coverage']
            if not isinstance(coverage, list) or len(coverage) > 100:
                raise ValueError('Invalid call history coverage')
            covered = set()
            message_ids = set()
            for entry in coverage:
                if not isinstance(entry, dict) or set(entry) != {'item_id', 'message_id', 'index'}:
                    raise ValueError('Invalid call history coverage')
                index = entry['index']
                message_id = entry['message_id']
                if (
                    type(index) is not int
                    or not 0 <= index < len(event['messages'])
                    or not isinstance(message_id, str)
                    or not 0 < len(message_id) <= 256
                ):
                    raise ValueError('Invalid saved speech identity')
                item = self.inputs.get(entry['item_id'])
                message = event['messages'][index]
                if entry['item_id'] in self.retired:
                    continue
                if not item or not item.digest or message['role'] != 'user':
                    raise ValueError('Unknown saved speech')
                if (
                    entry['item_id'] in covered
                    or message_id in message_ids
                    or item.message_id not in {None, message_id}
                ):
                    raise ValueError('Ambiguous saved speech identity')
                # Identity comes from call/item/message IDs, not matching utterance strings.
                # The digest only rejects truncated or transformed representations of that item.
                if text_digest(message['content']) != item.digest:
                    raise ValueError('Saved speech is not fully represented')
                item.message_id = message_id
                covered.add(entry['item_id'])
                message_ids.add(message_id)
            self.context = (commands[-1], covered)
            return []
        if kind == 'bridge.response.settled':
            if set(event) != {'type', 'response_id', 'continuation'} or type(event['continuation']) is not bool:
                raise ValueError('Invalid response settlement')
            response = self.responses.get(event['response_id'])
            if not response or not response['done']:
                raise ValueError('Response has not finished')
            response['settled'] = True
            response['continuation'] = event['continuation']
            return []
        if kind == 'bridge.cancel_pending':
            if set(event) != {'type', 'request_id'}:
                raise ValueError('Invalid pending response cancellation')
            # Cancellation acknowledgements are emitted by cancel_pending under the same writer lock.
            return []
        result_speech = None
        if kind == 'bridge.result':
            if type(event.get('speak')) is not bool:
                raise ValueError('Missing result continuation policy')
            result_speech = event['speak']
            event = {k: v for k, v in event.items() if k != 'speak'}
        generating = kind in {'bridge.respond', 'bridge.animation.respond', 'bridge.status'}
        request_id = event.get('request_id') if generating else None
        if generating:
            if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
                raise ValueError('Invalid response request identity')
            event = {k: v for k, v in event.items() if k != 'request_id'}
            if kind == 'bridge.respond' and 'item_id' in event:
                previous = self.inputs.get(event['item_id'])
                if event['item_id'] in self.retired or previous and (previous.superseded or previous.rejected):
                    self.cancelled_requests.append(request_id)
                    return []
            # Validate at dispatch: CallProtocol.command consumes one-shot response rights.
            if self.pending or self.sending:
                raise ValueError('A voice response is already requested')
            owner = None
            anchor = self.selected_order
            if kind == 'bridge.respond' and 'item_id' in event:
                owner = event['item_id']
                item = self.inputs.get(owner)
                if not item or not item.digest or item.rejected or item.superseded or item.assigned:
                    raise ValueError('Unknown or already answered input')
                item.assigned = True
                self.selected_order = item.order
                anchor = item.order
                for other_id, other in self.inputs.items():
                    if other.order < item.order:
                        other.superseded = True
                        log.debug('Input superseded: item=%s by=%s', safe_id(other_id), safe_id(owner))
            elif kind == 'bridge.respond' and 'call_id' in event:
                owner = self.native_input(self.call_inputs.get(event['call_id']))
                source = next(
                    (response for response in self.responses.values() if event['call_id'] in response['calls']), None
                )
                if source:
                    anchor = source['anchor']
            elif kind == 'bridge.animation.respond':
                source = self.responses.get(event.get('response_id'), {})
                owner = self.native_input(source.get('input'))
                anchor = source.get('anchor')
            if owner is None:
                # Superseded continuations speak task/text results into the
                # selected conversation, not a second answer to the old input.
                # Only selection rebases them; later commits still block dispatch.
                anchor = self.selected_order
            self.pending = (event, owner, request_id, anchor)
            self.wait_since = time.monotonic()
            return []
        command = self.protocol.command(event)
        if kind == 'bridge.result':
            self.protocol.requested.add(f'result:{event["call_id"]}')
            if result_speech:
                self.continuations.add(event['call_id'])
            else:
                self.continuations.discard(event['call_id'])
        return command if isinstance(command, list) else [command]

    async def cancel_pending(self, event, downstream):
        if self.pending and self.pending[2] == event['request_id']:
            _, owner, request_id, _ = self.pending
            if owner and owner in self.inputs:
                self.inputs[owner].superseded = True
            self.pending = None
            self.wait_since = None
            await downstream({'type': 'bridge.dispatch.cancelled', 'request_id': request_id})

    def observe(self, event):
        kind = event.get('type', '')
        if kind == 'input_audio_buffer.committed':
            self.input(event['item_id'])
        elif kind in {'conversation.item.added', 'conversation.item.done', 'conversation.item.created'}:
            item = event.get('item', {})
            if item.get('role') == 'user':
                audio_parts = sum(part.get('type') == 'input_audio' for part in item.get('content', []))
                if audio_parts:
                    # The microphone buffer commits one audio part per item. Do
                    # not let an unsupported multi-part item evade the prompt budget.
                    self.unsupported_audio_parts |= audio_parts != 1
                    self.input(item['id'])
        elif kind.startswith('conversation.item.input_audio_transcription.'):
            item = self.input(event['item_id'])
            if item:
                text = event.get('transcript', '').strip()
                if kind.endswith('.completed') and text:
                    item.digest = text_digest(text)
                    # Private frontend ordering hint; ASR completion order need not be speech order.
                    event['bridge_input_order'] = item.order
                elif kind.endswith(('.completed', '.failed')):
                    item.rejected = True
        elif kind == 'response.created':
            response = event['response']
            if response['id'] in self.responses:
                return
            metadata = response.get('metadata') or {}
            owner = metadata.get('input_item_id') or self.call_inputs.get(metadata.get('call_id'))
            audio = frozenset(self.inputs)
            anchor = self.selected_order
            if self.sending:
                owner = self.sending[1]
                audio = self.sending[3]
                anchor = self.sending[4]
                self.sending = None
            self.responses[response['id']] = {
                'input': owner,
                'audio': audio,
                'anchor': anchor,
                'calls': set(),
                'done': False,
                'settled': False,
                'continuation': False,
            }
        elif kind == 'response.done':
            response = self.responses.get(event['response']['id'])
            if response:
                response['done'] = True
        elif kind == 'response.output_item.done' and event.get('item', {}).get('type') == 'function_call':
            item = event['item']
            if item.get('status') == 'completed':
                response = self.responses.get(event['response_id'])
                self.call_inputs[item['call_id']] = response.get('input') if response else None
                if response:
                    response['calls'].add(item['call_id'])

        operation = self.operation
        if not operation:
            return
        target = operation['target']
        if (
            operation['kind'] == 'context'
            and kind in {'conversation.item.added', 'conversation.item.done', 'conversation.item.created'}
            and event.get('item', {}).get('id') == target
        ):
            self.remove_context = self.context_id
            self.context_id = target
            for item_id, item in self.inputs.items():
                # A historical acknowledgement proves persistence, not coverage
                # by the replacement snapshot used for today's retirement.
                item.covered = item_id in operation['coverage']
                item.established |= item.covered
            self.operation = None
        elif (
            operation['kind'] in {'retire', 'context_delete'}
            and kind == 'conversation.item.deleted'
            and event.get('item_id') == target
        ):
            if operation['kind'] == 'retire':
                self.inputs.pop(target, None)
                self.retired.add(target)
            self.operation = None
        else:
            return
        log.debug(
            'Input policy acknowledged: operation=%s event=%s item=%s retained=%d',
            operation['kind'],
            operation['event_id'],
            safe_id(target),
            len(self.inputs),
        )
        if self.pending:
            self.wait_since = time.monotonic()

    def protected_audio(self):
        protected = set(self.sending[3]) if self.sending else set()
        if self.pending and self.pending[1]:
            protected.add(self.pending[1])
        for response in self.responses.values():
            continuing = response['continuation'] or any(
                call_id in self.protocol.functions
                or call_id in self.protocol.animation_calls
                or call_id in self.continuations
                for call_id in response['calls']
            )
            # All window members, not just the owner, were consumed by this
            # request. Supersession releases continuation rights, never live generation.
            if (
                not response['done']
                or not response['settled']
                or continuing
                and response['anchor'] == self.selected_order
            ):
                protected.update(response['audio'])
            else:
                response['audio'] = frozenset()
        return protected

    async def fail(self, downstream, reason):
        if self.blocked:
            return
        self.blocked = True
        operation = self.operation or {}
        log.warning(
            'Input policy blocked: reason=%s operation=%s event=%s item=%s revision=%s',
            reason,
            operation.get('kind'),
            operation.get('event_id'),
            safe_id(operation.get('target', '')),
            self.protocol.context_revision,
        )
        await downstream(
            {
                'type': 'bridge.error',
                'fatal': False,
                'code': 'input_audio_sync',
                'message': 'Input audio synchronization failed. No further answers will be requested in this call. '
                'Your saved chat is unchanged. End the call and start a new one; check transcription '
                'coverage and provider conversation acknowledgements if this repeats.',
            }
        )
        self.pending = None

    async def provider_error(self, event, downstream):
        # Only an exact cleanup event correlation can bypass the existing fatal provider-error policy.
        if self.operation and event.get('error', {}).get('event_id') == self.operation['event_id']:
            await self.fail(downstream, 'provider_rejected_operation')
            return True
        return False

    async def start_operation(self, kind, command, upstream, coverage=()):
        event_id = f'owui_cleanup_{uuid4().hex}'
        target = command.get('item_id') or command['item']['id']
        self.operation = {
            'kind': kind,
            'event_id': event_id,
            'target': target,
            'coverage': coverage,
            'deadline': time.monotonic() + SYNC_TIMEOUT,
        }
        if self.pending:
            self.wait_since = time.monotonic()
        log.debug(
            'Input policy requested: operation=%s event=%s item=%s revision=%s retained=%d',
            kind,
            event_id,
            safe_id(target),
            self.protocol.context_revision,
            len(self.inputs),
        )
        await upstream({**command, 'event_id': event_id})

    async def pump(self, upstream, downstream):
        for request_id in self.cancelled_requests:
            await downstream({'type': 'bridge.dispatch.cancelled', 'request_id': request_id})
        self.cancelled_requests.clear()
        if self.blocked:
            return
        if self.unsupported_audio_parts:
            await self.fail(downstream, 'multiple_audio_parts_in_microphone_item')
            return
        now = time.monotonic()
        if self.sending and now >= self.sending[2]:
            await self.fail(downstream, 'response_creation_timeout')
            return
        if (
            not self.operation
            and self.pending
            and self.wait_since is not None
            and now - self.wait_since >= DEPENDENCY_TIMEOUT
        ):
            await self.fail(downstream, 'unresolved_input_or_text_coverage')
            return
        if self.operation:
            if now >= self.operation['deadline']:
                await self.fail(downstream, 'acknowledgement_timeout')
            return
        if self.remove_context:
            target, self.remove_context = self.remove_context, None
            await self.start_operation(
                'context_delete', {'type': 'conversation.item.delete', 'item_id': target}, upstream
            )
            return
        if self.context:
            (command, coverage), self.context = self.context, None
            await self.start_operation('context', command, upstream, coverage)
            return
        window = self.recent_window(self.selected_order)
        protected = self.protected_audio()
        missing_coverage = []
        for item_id, item in self.inputs.items():
            # Count one retains the verified eager-release behavior. Larger
            # windows keep settled acoustic history between turns, even if its
            # right to request a direct answer has been superseded.
            outside_window = (
                self.retention_count == 1
                or self.selected_order is not None
                and item.order <= self.selected_order
                and item_id not in window
            )
            eligible = item.rejected or (item.assigned or item.superseded) and outside_window
            if not eligible or item_id in protected:
                continue
            if not item.rejected and not item.covered:
                missing_coverage.append(item_id)
                continue
            log.debug(
                'Input eligible: item=%s reason=%s window=%d retained=%d',
                safe_id(item_id),
                'rejected' if item.rejected else 'out_of_window' if self.retention_count > 1 else 'completed',
                self.retention_count,
                len(self.inputs),
            )
            await self.start_operation('retire', {'type': 'conversation.item.delete', 'item_id': item_id}, upstream)
            return
        if self.pending:
            event, owner, request_id, anchor = self.pending
            # Direct/native continuations retain their turn boundary; superseded
            # task results use the selected conversation without restoring an owner.
            permitted = self.recent_window(anchor) if self.retention_count > 1 else {owner}
            # Historical members may coexist, but unresolved/future items must
            # not leak into the selected response just because there is room.
            dependencies = (
                ()
                if event['type'] == 'bridge.status'
                else tuple(
                    sorted(
                        item_id
                        for item_id, item in self.inputs.items()
                        if item_id not in permitted
                        or item.rejected
                        or not item.digest
                        or item_id != owner
                        and not item.established
                    )
                )
            )
            active = self.sending or any(not response['done'] for response in self.responses.values())
            within_budget = event['type'] == 'bridge.status' or len(self.inputs) <= self.retention_count
            if not dependencies and not active and not missing_coverage and within_budget:
                command = self.protocol.command(event)
                if event['type'] == 'bridge.respond' and 'call_id' in event:
                    self.continuations.discard(event['call_id'])
                if event['type'] == 'bridge.animation.respond':
                    self.responses[event['response_id']]['continuation'] = False
                    # A late animation may continue as text/task context, never by resurrecting audio.
                    if not owner:
                        command['response']['metadata'].pop('input_item_id', None)
                audio = frozenset() if event['type'] == 'bridge.status' else frozenset(self.inputs)
                self.sending = (request_id, owner, now + SYNC_TIMEOUT, audio, anchor)
                self.pending = None
                self.wait_since = None
                await upstream(command)
                return
            if dependencies != self.last_dependencies:
                log.debug(
                    'Input response deferred: request=%s selected=%s dependencies=%s retained=%d window=%d protected=%s uncovered=%s',
                    safe_id(request_id),
                    safe_id(owner),
                    [safe_id(item) for item in dependencies],
                    len(self.inputs),
                    self.retention_count,
                    [safe_id(item) for item in sorted(protected)],
                    [safe_id(item) for item in missing_coverage],
                )
                self.last_dependencies = dependencies
        if self.pending or missing_coverage:
            if self.wait_since is None:
                self.wait_since = now
            elif now - self.wait_since >= DEPENDENCY_TIMEOUT:
                await self.fail(downstream, 'unresolved_input_or_text_coverage')
        else:
            self.wait_since = None
