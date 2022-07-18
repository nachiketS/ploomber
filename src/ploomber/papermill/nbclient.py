import asyncio
import typing as t

from nbclient import NotebookClient
# NOTE: run_hook was added recently, we should pin an appropriate version
from nbclient.util import ensure_async, run_hook, run_sync
from nbclient.exceptions import CellControlSignal, DeadKernelError
from nbformat import NotebookNode
from jupyter_client.client import KernelClient
from nbclient.client import timestamp


class PloomberNotebookClient2(NotebookClient):
    async def async_start_new_kernel_client(self) -> KernelClient:
        """Creates a new kernel client.

        Returns
        -------
        kc : KernelClient
            Kernel client as created by the kernel manager ``km``.
        """
        assert self.km is not None
        try:
            self.kc = self.km.client()
            await ensure_async(self.kc.start_channels()
                               )  # type:ignore[func-returns-value]
            await ensure_async(
                self.kc.wait_for_ready(timeout=self.startup_timeout)
            )  # type:ignore
        except Exception as e:
            self.log.error(
                "Error occurred while starting new kernel client for kernel {}: {}"
                .format(self.km.kernel_id, str(e)))
            await self._async_cleanup_kernel()
            raise

        # ENABLE STDIN
        self.kc.allow_stdin = True
        await run_hook(self.on_notebook_start, notebook=self.nb)
        return self.kc

    start_new_kernel_client = run_sync(async_start_new_kernel_client)

    async def _async_poll_stdin_msg(self, parent_msg_id: str,
                                    cell: NotebookNode,
                                    cell_index: int) -> None:
        assert self.kc is not None

        # looks like this sometimes returns false
        # msg_received = await self.kc.stdin_channel.msg_ready()

        if True:
            # if msg_received:  #or cell_index == 0:
            from queue import Empty

            while True:
                try:
                    msg = await ensure_async(
                        self.kc.stdin_channel.get_msg(timeout=0.1))
                except Empty:
                    break
                else:
                    # flush io
                    flush_io(self.kc)

                    self.kc.input(input(msg['content']['prompt']))

                    try:
                        msg = await self.kc.get_iopub_msg(timeout=1)
                    except Empty:
                        pass
                    else:
                        if msg.get('content', {}).get('text'):
                            print(msg['content']['text'])
                        else:
                            print(msg['content'])

    async def async_execute_cell(
        self,
        cell: NotebookNode,
        cell_index: int,
        execution_count: t.Optional[int] = None,
        store_history: bool = True,
    ) -> NotebookNode:
        """
        Executes a single code cell.

        To execute all cells see :meth:`execute`.

        Parameters
        ----------
        cell : nbformat.NotebookNode
            The cell which is currently being processed.
        cell_index : int
            The position of the cell within the notebook object.
        execution_count : int
            The execution count to be assigned to the cell (default: Use kernel response)
        store_history : bool
            Determines if history should be stored in the kernel (default: False).
            Specific to ipython kernels, which can store command histories.

        Returns
        -------
        output : dict
            The execution output payload (or None for no output).

        Raises
        ------
        CellExecutionError
            If execution failed and should raise an exception, this will be raised
            with defaults about the failure.

        Returns
        -------
        cell : NotebookNode
            The cell which was just processed.
        """
        assert self.kc is not None

        await run_hook(self.on_cell_start, cell=cell, cell_index=cell_index)

        if cell.cell_type != 'code' or not cell.source.strip():
            self.log.debug("Skipping non-executing cell %s", cell_index)
            return cell

        if self.skip_cells_with_tag in cell.metadata.get("tags", []):
            self.log.debug("Skipping tagged cell %s", cell_index)
            return cell

        if self.record_timing:  # clear execution metadata prior to execution
            cell['metadata']['execution'] = {}

        self.log.debug("Executing cell:\n%s", cell.source)

        cell_allows_errors = (not self.force_raise_errors) and (
            self.allow_errors
            or "raises-exception" in cell.metadata.get("tags", []))

        await run_hook(self.on_cell_execute, cell=cell, cell_index=cell_index)
        # execute cell
        parent_msg_id = await ensure_async(
            self.kc.execute(cell.source,
                            store_history=store_history,
                            stop_on_error=not cell_allows_errors))
        await run_hook(self.on_cell_complete, cell=cell, cell_index=cell_index)

        # important: first thing we do is to check for stdin
        import time
        # NOTE: this is needed when starting entering %pdb but not in input
        time.sleep(2)
        await self._async_poll_stdin_msg(parent_msg_id, cell, cell_index)

        # We launched a code cell to execute
        self.code_cells_executed += 1
        exec_timeout = self._get_timeout(cell)

        cell.outputs = []
        self.clear_before_next_output = False

        task_poll_kernel_alive = asyncio.ensure_future(
            self._async_poll_kernel_alive())

        # poll output msg (iopub channel)
        task_poll_output_msg = asyncio.ensure_future(
            self._async_poll_output_msg(parent_msg_id, cell, cell_index))
        # poll reply (shell channel)
        # NOTE: this is passing task_poll_output_msg as argument!
        self.task_poll_for_reply = asyncio.ensure_future(
            self._async_poll_for_reply(parent_msg_id, cell, exec_timeout,
                                       task_poll_output_msg,
                                       task_poll_kernel_alive))

        # if cell_index == 5:
        #     from ipdb import set_trace
        #     set_trace()
        try:
            exec_reply = await self.task_poll_for_reply
        except asyncio.CancelledError:
            # can only be cancelled by task_poll_kernel_alive when the kernel is dead
            task_poll_output_msg.cancel()
            raise DeadKernelError("Kernel died")
        except Exception as e:
            # Best effort to cancel request if it hasn't been resolved
            try:
                # Check if the task_poll_output is doing the raising for us
                if not isinstance(e, CellControlSignal):
                    task_poll_output_msg.cancel()
            finally:
                raise

        if execution_count:
            cell['execution_count'] = execution_count
        await run_hook(self.on_cell_executed,
                       cell=cell,
                       cell_index=cell_index,
                       execute_reply=exec_reply)

        await self._check_raise_for_error(cell, cell_index, exec_reply)

        self.nb['cells'][cell_index] = cell

        return cell


# taken from
# https://github.com/jupyter/jupyter_console/blob/bcf17a3953844d75262e3ce23b784832d9044877/jupyter_console/ptshell.py#L846
def flush_io(client):
    import sys

    while run_sync(client.iopub_channel.msg_ready)():
        sub_msg = run_sync(client.iopub_channel.get_msg)()
        msg_type = sub_msg['header']['msg_type']

        _pending_clearoutput = True

        # Update execution_count in case it changed in another session
        # if msg_type == "execute_input":
        #     self.execution_count = int(
        #         sub_msg["content"]["execution_count"]) + 1

        if True:  #self.include_output(sub_msg):
            if msg_type == 'status':
                # self._execution_state = sub_msg["content"]["execution_state"]
                pass

            elif msg_type == 'stream':
                if sub_msg["content"]["name"] == "stdout":
                    if _pending_clearoutput:
                        print("\r", end="")
                        _pending_clearoutput = False
                    print(sub_msg["content"]["text"], end="")
                    sys.stdout.flush()
                elif sub_msg["content"]["name"] == "stderr":
                    if _pending_clearoutput:
                        print("\r", file=sys.stderr, end="")
                        _pending_clearoutput = False
                    print(sub_msg["content"]["text"], file=sys.stderr, end="")
                    sys.stderr.flush()

            elif msg_type == 'execute_result':
                if _pending_clearoutput:
                    print("\r", end="")
                    _pending_clearoutput = False
                # self.execution_count = int(
                #     sub_msg["content"]["execution_count"])
                # if not self.from_here(sub_msg):
                # sys.stdout.write(self.other_output_prefix)
                format_dict = sub_msg["content"]["data"]
                # self.handle_rich_data(format_dict)

                if 'text/plain' not in format_dict:
                    continue

                # prompt_toolkit writes the prompt at a slightly lower level,
                # so flush streams first to ensure correct ordering.
                sys.stdout.flush()
                sys.stderr.flush()
                # self.print_out_prompt()
                text_repr = format_dict['text/plain']
                if '\n' in text_repr:
                    # For multi-line results, start a new line after prompt
                    print()
                print(text_repr)

                # Remote: add new prompt
                # if not self.from_here(sub_msg):
                #     sys.stdout.write('\n')
                #     sys.stdout.flush()
                #     self.print_remote_prompt()

            elif msg_type == 'display_data':
                data = sub_msg["content"]["data"]
                # handled = self.handle_rich_data(data)
                # if not handled:
                #     if not self.from_here(sub_msg):
                #         sys.stdout.write(self.other_output_prefix)
                #     # if it was an image, we handled it by now
                #     if 'text/plain' in data:
                #         print(data['text/plain'])

            # If execute input: print it
            elif msg_type == 'execute_input':
                content = sub_msg['content']
                # ec = content.get('execution_count', self.execution_count - 1)

                # New line
                sys.stdout.write('\n')
                sys.stdout.flush()

                # With `Remote In [3]: `
                # self.print_remote_prompt(ec=ec)

                # And the code
                sys.stdout.write(content['code'] + '\n')

            elif msg_type == 'clear_output':
                if sub_msg["content"]["wait"]:
                    _pending_clearoutput = True
                else:
                    print("\r", end="")

            elif msg_type == 'error':
                for frame in sub_msg["content"]["traceback"]:
                    print(frame, file=sys.stderr)


import asyncio
import atexit
import base64
import collections
import datetime
import signal
import typing as t
from contextlib import asynccontextmanager, contextmanager
from queue import Empty
from textwrap import dedent
from time import monotonic

from jupyter_client import KernelManager
from jupyter_client.client import KernelClient
from nbformat import NotebookNode
from nbformat.v4 import output_from_msg
from traitlets import (
    Any,
    Bool,
    Callable,
    Dict,
    Enum,
    Integer,
    List,
    Type,
    Unicode,
    default,
)
from traitlets.config.configurable import LoggingConfigurable

from nbclient.exceptions import (
    CellControlSignal,
    CellExecutionComplete,
    CellExecutionError,
    CellTimeoutError,
    DeadKernelError,
)
from nbclient.output_widget import OutputWidget
from nbclient.util import ensure_async, run_hook, run_sync


class PloomberNotebookClient(NotebookClient):
    """
    Encompasses a Client for executing cells in a notebook
    """
    async def async_start_new_kernel_client(self) -> KernelClient:
        """Creates a new kernel client.

        Returns
        -------
        kc : KernelClient
            Kernel client as created by the kernel manager ``km``.
        """
        assert self.km is not None
        try:
            self.kc = self.km.client()
            await ensure_async(self.kc.start_channels()
                               )  # type:ignore[func-returns-value]
            await ensure_async(
                self.kc.wait_for_ready(timeout=self.startup_timeout)
            )  # type:ignore
        except Exception as e:
            self.log.error(
                "Error occurred while starting new kernel client for kernel {}: {}"
                .format(self.km.kernel_id, str(e)))
            await self._async_cleanup_kernel()
            raise
        self.kc.allow_stdin = True
        await run_hook(self.on_notebook_start, notebook=self.nb)
        return self.kc

    start_new_kernel_client = run_sync(async_start_new_kernel_client)

    async def _async_poll_stdin_msg(self, parent_msg_id: str,
                                    cell: NotebookNode,
                                    cell_index: int) -> None:
        print(f'enter stdin polling!')
        assert self.kc is not None

        # looks like this sometimes returns false
        # msg_received = await self.kc.stdin_channel.msg_ready()

        if True:
            # if msg_received:  #or cell_index == 0:
            print(f'stdin message ready!')
            from queue import Empty

            while True:
                try:
                    msg = await ensure_async(
                        self.kc.stdin_channel.get_msg(timeout=0.1))
                except Empty:
                    break
                else:
                    # flush io
                    flush_io(self.kc)

                    self.kc.input(input(msg['content']['prompt']))

                    try:
                        msg = await self.kc.get_iopub_msg(timeout=1)
                    except Empty:
                        pass
                    else:
                        if msg.get('content', {}).get('text'):
                            print(msg['content']['text'])
                        else:
                            print(msg['content'])
        else:
            print('no message!')

    async def _async_poll_kernel_alive(self) -> None:
        while True:
            await asyncio.sleep(1)
            try:
                await self._async_check_alive()
            except DeadKernelError:
                assert self.task_poll_for_reply is not None
                self.task_poll_for_reply.cancel()
                return

    def _get_timeout(self, cell: t.Optional[NotebookNode]) -> t.Optional[int]:
        if self.timeout_func is not None and cell is not None:
            timeout = self.timeout_func(cell)
        else:
            timeout = self.timeout

        if not timeout or timeout < 0:
            timeout = None

        return timeout

    async def _async_handle_timeout(
            self,
            timeout: int,
            cell: t.Optional[NotebookNode] = None) -> t.Union[None, t.Dict]:

        self.log.error("Timeout waiting for execute reply (%is)." % timeout)
        if self.interrupt_on_timeout:
            self.log.error("Interrupting kernel")
            assert self.km is not None
            await ensure_async(self.km.interrupt_kernel())
            if self.error_on_timeout:
                execute_reply = {
                    "content": {
                        **self.error_on_timeout, "status": "error"
                    }
                }
                return execute_reply
            return None
        else:
            assert cell is not None
            raise CellTimeoutError.error_from_timeout_and_cell(
                "Cell execution timed out", timeout, cell)

    async def _async_check_alive(self) -> None:
        assert self.kc is not None
        if not await ensure_async(self.kc.is_alive()):  # type:ignore
            self.log.error("Kernel died while waiting for execute reply.")
            raise DeadKernelError("Kernel died")

    async def async_wait_for_reply(
            self,
            msg_id: str,
            cell: t.Optional[NotebookNode] = None) -> t.Optional[t.Dict]:

        assert self.kc is not None
        # wait for finish, with timeout
        timeout = self._get_timeout(cell)
        cummulative_time = 0
        while True:
            try:
                msg: t.Dict = await ensure_async(
                    self.kc.shell_channel.get_msg(
                        timeout=self.shell_timeout_interval))
            except Empty:
                await self._async_check_alive()
                cummulative_time += self.shell_timeout_interval
                if timeout and cummulative_time > timeout:
                    await self._async_handle_timeout(timeout, cell)
                    break
            else:
                if msg['parent_header'].get('msg_id') == msg_id:
                    return msg
        return None

    wait_for_reply = run_sync(async_wait_for_reply)
    # Backwards compatibility naming for papermill
    _wait_for_reply = wait_for_reply

    def _passed_deadline(self, deadline: int) -> bool:
        if deadline is not None and deadline - monotonic() <= 0:
            return True
        return False

    async def _check_raise_for_error(self, cell: NotebookNode, cell_index: int,
                                     exec_reply: t.Optional[t.Dict]) -> None:

        if exec_reply is None:
            return None

        exec_reply_content = exec_reply['content']
        if exec_reply_content['status'] != 'error':
            return None

        cell_allows_errors = (not self.force_raise_errors) and (
            self.allow_errors
            or exec_reply_content.get('ename') in self.allow_error_names
            or "raises-exception" in cell.metadata.get("tags", []))
        await run_hook(self.on_cell_error,
                       cell=cell,
                       cell_index=cell_index,
                       execute_reply=exec_reply)
        if not cell_allows_errors:
            raise CellExecutionError.from_cell_and_msg(cell,
                                                       exec_reply_content)

    async def async_execute_cell(
        self,
        cell: NotebookNode,
        cell_index: int,
        execution_count: t.Optional[int] = None,
        store_history: bool = True,
    ) -> NotebookNode:
        """
        Executes a single code cell.

        To execute all cells see :meth:`execute`.

        Parameters
        ----------
        cell : nbformat.NotebookNode
            The cell which is currently being processed.
        cell_index : int
            The position of the cell within the notebook object.
        execution_count : int
            The execution count to be assigned to the cell (default: Use kernel response)
        store_history : bool
            Determines if history should be stored in the kernel (default: False).
            Specific to ipython kernels, which can store command histories.

        Returns
        -------
        output : dict
            The execution output payload (or None for no output).

        Raises
        ------
        CellExecutionError
            If execution failed and should raise an exception, this will be raised
            with defaults about the failure.

        Returns
        -------
        cell : NotebookNode
            The cell which was just processed.
        """
        assert self.kc is not None

        await run_hook(self.on_cell_start, cell=cell, cell_index=cell_index)

        if cell.cell_type != 'code' or not cell.source.strip():
            self.log.debug("Skipping non-executing cell %s", cell_index)
            return cell

        if self.skip_cells_with_tag in cell.metadata.get("tags", []):
            self.log.debug("Skipping tagged cell %s", cell_index)
            return cell

        if self.record_timing:  # clear execution metadata prior to execution
            cell['metadata']['execution'] = {}

        self.log.debug("Executing cell:\n%s", cell.source)

        cell_allows_errors = (not self.force_raise_errors) and (
            self.allow_errors
            or "raises-exception" in cell.metadata.get("tags", []))

        await run_hook(self.on_cell_execute, cell=cell, cell_index=cell_index)
        # execute cell
        print(f'EXECUTING CELL {cell_index} - {cell["source"]}')
        parent_msg_id = await ensure_async(
            self.kc.execute(cell.source,
                            store_history=store_history,
                            stop_on_error=not cell_allows_errors))
        await run_hook(self.on_cell_complete, cell=cell, cell_index=cell_index)

        # important: first thing we do is to check for stdin
        print(f'polling stdin message (idx {cell_index})')
        # from ipdb import set_trace
        # set_trace()
        import time
        # NOTE: this is needed when starting entering %pdb but not in input
        time.sleep(2)
        await self._async_poll_stdin_msg(parent_msg_id, cell, cell_index)
        # time.sleep(2)
        print('finished polling stdin message')

        # We launched a code cell to execute
        self.code_cells_executed += 1
        exec_timeout = self._get_timeout(cell)

        cell.outputs = []
        self.clear_before_next_output = False

        task_poll_kernel_alive = asyncio.ensure_future(
            self._async_poll_kernel_alive())

        # poll output msg (iopub channel)
        task_poll_output_msg = asyncio.ensure_future(
            self._async_poll_output_msg(parent_msg_id, cell, cell_index))
        # poll reply (shell channel)
        # NOTE: this is passing task_poll_output_msg as argument!
        self.task_poll_for_reply = asyncio.ensure_future(
            self._async_poll_for_reply(parent_msg_id, cell, exec_timeout,
                                       task_poll_output_msg,
                                       task_poll_kernel_alive))

        print(f'poll for reply {cell_index}')
        # if cell_index == 5:
        #     from ipdb import set_trace
        #     set_trace()
        try:
            exec_reply = await self.task_poll_for_reply
        except asyncio.CancelledError:
            # can only be cancelled by task_poll_kernel_alive when the kernel is dead
            task_poll_output_msg.cancel()
            raise DeadKernelError("Kernel died")
        except Exception as e:
            # Best effort to cancel request if it hasn't been resolved
            try:
                # Check if the task_poll_output is doing the raising for us
                if not isinstance(e, CellControlSignal):
                    task_poll_output_msg.cancel()
            finally:
                raise
        print('end polling')

        if execution_count:
            cell['execution_count'] = execution_count
        await run_hook(self.on_cell_executed,
                       cell=cell,
                       cell_index=cell_index,
                       execute_reply=exec_reply)
        print('checking error')
        await self._check_raise_for_error(cell, cell_index, exec_reply)
        print('checked error')
        self.nb['cells'][cell_index] = cell
        print(f'RETURNING CELL {cell_index}')
        return cell

    execute_cell = run_sync(async_execute_cell)

    def process_message(self, msg: t.Dict, cell: NotebookNode,
                        cell_index: int) -> t.Optional[NotebookNode]:
        """
        Processes a kernel message, updates cell state, and returns the
        resulting output object that was appended to cell.outputs.

        The input argument *cell* is modified in-place.

        Parameters
        ----------
        msg : dict
            The kernel message being processed.
        cell : nbformat.NotebookNode
            The cell which is currently being processed.
        cell_index : int
            The position of the cell within the notebook object.

        Returns
        -------
        output : NotebookNode
            The execution output payload (or None for no output).

        Raises
        ------
        CellExecutionComplete
          Once a message arrives which indicates computation completeness.

        """
        msg_type = msg['msg_type']
        self.log.debug("msg_type: %s", msg_type)
        content = msg['content']
        self.log.debug("content: %s", content)

        display_id = content.get('transient', {}).get('display_id', None)
        if display_id and msg_type in {
                'execute_result', 'display_data', 'update_display_data'
        }:
            self._update_display_id(display_id, msg)

        # set the prompt number for the input and the output
        if 'execution_count' in content:
            cell['execution_count'] = content['execution_count']

        if self.record_timing:
            if msg_type == 'status':
                if content['execution_state'] == 'idle':
                    cell['metadata']['execution'][
                        'iopub.status.idle'] = timestamp(msg)
                elif content['execution_state'] == 'busy':
                    cell['metadata']['execution'][
                        'iopub.status.busy'] = timestamp(msg)
            elif msg_type == 'execute_input':
                cell['metadata']['execution'][
                    'iopub.execute_input'] = timestamp(msg)

        if msg_type == 'status':
            if content['execution_state'] == 'idle':
                raise CellExecutionComplete()
        elif msg_type == 'clear_output':
            self.clear_output(cell.outputs, msg, cell_index)
        elif msg_type.startswith('comm'):
            self.handle_comm_msg(cell.outputs, msg, cell_index)
        # Check for remaining messages we don't process
        elif msg_type not in ['execute_input', 'update_display_data']:
            # Assign output as our processed "result"
            return self.output(cell.outputs, msg, display_id, cell_index)
        return None

    def output(self, outs: t.List, msg: t.Dict, display_id: str,
               cell_index: int) -> t.Optional[NotebookNode]:

        msg_type = msg['msg_type']
        out = None

        parent_msg_id = msg['parent_header'].get('msg_id')
        if self.output_hook_stack[parent_msg_id]:
            # if we have a hook registered, it will override our
            # default output behaviour (e.g. OutputWidget)
            hook = self.output_hook_stack[parent_msg_id][-1]
            hook.output(outs, msg, display_id, cell_index)
            return None

        try:
            out = output_from_msg(msg)
        except ValueError:
            self.log.error(f"unhandled iopub msg: {msg_type}")
            return None

        if self.clear_before_next_output:
            self.log.debug('Executing delayed clear_output')
            outs[:] = []
            self.clear_display_id_mapping(cell_index)
            self.clear_before_next_output = False

        if display_id:
            # record output index in:
            #   _display_id_map[display_id][cell_idx]
            cell_map = self._display_id_map.setdefault(display_id, {})
            output_idx_list = cell_map.setdefault(cell_index, [])
            output_idx_list.append(len(outs))

        outs.append(out)

        return out  # type:ignore[no-any-return]

    def clear_output(self, outs: t.List, msg: t.Dict, cell_index: int) -> None:

        content = msg['content']

        parent_msg_id = msg['parent_header'].get('msg_id')
        if self.output_hook_stack[parent_msg_id]:
            # if we have a hook registered, it will override our
            # default clear_output behaviour (e.g. OutputWidget)
            hook = self.output_hook_stack[parent_msg_id][-1]
            hook.clear_output(outs, msg, cell_index)
            return

        if content.get('wait'):
            self.log.debug('Wait to clear output')
            self.clear_before_next_output = True
        else:
            self.log.debug('Immediate clear output')
            outs[:] = []
            self.clear_display_id_mapping(cell_index)

    def clear_display_id_mapping(self, cell_index: int) -> None:

        for _, cell_map in self._display_id_map.items():
            if cell_index in cell_map:
                cell_map[cell_index] = []

    def handle_comm_msg(self, outs: t.List, msg: t.Dict,
                        cell_index: int) -> None:

        content = msg['content']
        data = content['data']
        if self.store_widget_state and 'state' in data:  # ignore custom msg'es
            self.widget_state.setdefault(content['comm_id'],
                                         {}).update(data['state'])
            if 'buffer_paths' in data and data['buffer_paths']:
                comm_id = content['comm_id']
                if comm_id not in self.widget_buffers:
                    self.widget_buffers[comm_id] = {}
                # for each comm, the path uniquely identifies a buffer
                new_buffers: t.Dict[t.Tuple[str, ...], t.Dict[str, str]] = {
                    tuple(k["path"]): k
                    for k in self._get_buffer_data(msg)
                }
                self.widget_buffers[comm_id].update(new_buffers)
        # There are cases where we need to mimic a frontend, to get similar behaviour as
        # when using the Output widget from Jupyter lab/notebook
        if msg['msg_type'] == 'comm_open':
            target = msg['content'].get('target_name')
            handler = self.comm_open_handlers.get(target)
            if handler:
                comm_id = msg['content']['comm_id']
                comm_object = handler(msg)
                if comm_object:
                    self.comm_objects[comm_id] = comm_object
            else:
                self.log.warning(
                    f'No handler found for comm target {target!r}')
        elif msg['msg_type'] == 'comm_msg':
            content = msg['content']
            comm_id = msg['content']['comm_id']
            if comm_id in self.comm_objects:
                self.comm_objects[comm_id].handle_msg(msg)

    def _serialize_widget_state(self, state: t.Dict) -> t.Dict[str, t.Any]:
        """Serialize a widget state, following format in @jupyter-widgets/schema."""
        return {
            'model_name': state.get('_model_name'),
            'model_module': state.get('_model_module'),
            'model_module_version': state.get('_model_module_version'),
            'state': state,
        }

    def _get_buffer_data(self, msg: t.Dict) -> t.List[t.Dict[str, str]]:
        encoded_buffers = []
        paths = msg['content']['data']['buffer_paths']
        buffers = msg['buffers']
        for path, buffer in zip(paths, buffers):
            encoded_buffers.append({
                'data':
                base64.b64encode(buffer).decode('utf-8'),
                'encoding':
                'base64',
                'path':
                path,
            })
        return encoded_buffers

    def register_output_hook(self, msg_id: str, hook: OutputWidget) -> None:
        """Registers an override object that handles output/clear_output instead.

        Multiple hooks can be registered, where the last one will be used (stack based)
        """
        # mimics
        # https://jupyterlab.github.io/jupyterlab/services/interfaces/kernel.ikernelconnection.html#registermessagehook
        self.output_hook_stack[msg_id].append(hook)

    def remove_output_hook(self, msg_id: str, hook: OutputWidget) -> None:
        """Unregisters an override object that handles output/clear_output instead"""
        # mimics
        # https://jupyterlab.github.io/jupyterlab/services/interfaces/kernel.ikernelconnection.html#removemessagehook
        removed_hook = self.output_hook_stack[msg_id].pop()
        assert removed_hook == hook

    def on_comm_open_jupyter_widget(self, msg: t.Dict) -> t.Optional[t.Any]:
        content = msg['content']
        data = content['data']
        state = data['state']
        comm_id = msg['content']['comm_id']
        module = self.widget_registry.get(state['_model_module'])
        if module:
            widget_class = module.get(state['_model_name'])
            if widget_class:
                return widget_class(comm_id, state, self.kc, self)
        return None
