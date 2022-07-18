import asyncio
import typing as t

from nbclient import NotebookClient
# NOTE: run_hook was added recently, we should pin an appropriate version
from nbclient.util import ensure_async, run_hook, run_sync
from nbclient.exceptions import CellControlSignal, DeadKernelError
from nbformat import NotebookNode
from jupyter_client.client import KernelClient
from nbclient.client import timestamp


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
