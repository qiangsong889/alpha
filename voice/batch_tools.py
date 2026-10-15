"""新批次流程的模型控制工具；参数不能直接授权外部执行。"""

from voice.plan_tools import CONTROL_TOOLS as PLAN_TOOLS, _tool

CONTROL_TOOLS = dict(PLAN_TOOLS)
CONTROL_TOOLS.update({
    "finish_collection": _tool(
        "每句用户输入整理完所有新增、修改、取消和顺序后必须调用。complete 表示整条任务链是否完整。"
        "等等、还有、再把等未完成表达应为 false，并给出具体 question。"
        "只有用户明确说执行吧才可 execute_now=true；不能借此绕过确认。",
        {"complete": {"type": "boolean"}, "question": {"type": "string"},
         "execute_now": {"type": "boolean"}}, ["complete"],
    ),
    "set_task_dependencies": _tool(
        "用户要求先后顺序时设置依赖。task_id 必须等 depends_on 中各项成功才能执行。",
        {"task_id": {"type": "integer", "minimum": 1},
         "depends_on": {"type": "array", "items": {"type": "integer", "minimum": 1}}},
        ["task_id", "depends_on"],
    ),
    "cancel_task": _tool(
        "用户取消前面的某项任务时使用固定 task_id。未执行则取消，已执行则登记精确撤销并等待确认。"
        "有多个同名目标时先澄清。created_ids 可选择本次回执内要撤销的部分任务 ID。",
        {"task_id": {"type": "integer", "minimum": 1},
         "created_ids": {"type": "array", "items": {"type": "string"}}}, ["task_id"],
    ),
    "cancel_remaining": _tool("用户明确要求停下或取消剩下的时使用，已发出的请求无法保证撤销。", {}, []),
    "verify_operation": _tool(
        "操作结果未知时先核实，不要重新添加或创建。operation_id 使用快照中的 operation_id。",
        {"operation_id": {"type": "string"}}, ["operation_id"],
    ),
})

BATCH_RULES = """
任务按用户原话的先后整理，可边说边修改。所有写工具只登记，不能声称已执行。
即使开关灯也要先登记；不要为了取消或改口直接做相反的设备操作。
每次新输入先看任务快照：替换用 update_plan_item，取消用 cancel_task，先后顺序用 set_task_dependencies。
用户说“不要牛奶了，加酱油”时先取消指定的前一项，再登记酱油；不能取消后面才新增的任务。
同一句话可以修改多处；修改完成后调用 finish_collection。信息不完整或用户说等等时 complete=false。
所有任务、依赖和修改都整理完后必须调用 finish_collection；不能因只调用了一个写工具就当作整理结束。
用户明确说“执行吧”时 finish_collection 的 execute_now=true，但不能绕过缺少参数或确认要求。
一到两个普通动作在静默两秒后提交；三个及以上或撤销操作由系统复述，用户明确同意后才提交。
confirm_plan 只表示提交排队，不表示已经执行成功。成功只能根据系统执行结果报告。
执行期间继续记录下一批，不主动插话。结果未知时 verify_operation，查不到不能宣称没发生或重试。
遇到不完整的任务，等待系统五秒后的澄清指令，不要反复催问。
系统要求复述时逐字念，不调用工具；用户提问、要求修改或含糊时不要确认。
"""


async def call_batch_control(workflow, name, arguments):
    import jsonschema  # type: ignore[import-untyped]
    jsonschema.validate(arguments, CONTROL_TOOLS[name]["parameters"])
    if name == "finish_collection":
        return workflow.finish_collection(**arguments)
    if name == "set_task_dependencies":
        return workflow.depend(**arguments)
    if name == "cancel_task":
        return await workflow.undo(arguments["task_id"], arguments.get("created_ids"))
    if name == "verify_operation":
        if error := workflow._input_error():
            return workflow.result(False, error)
        return await workflow.verify_operation(**arguments)
    if name == "cancel_remaining":
        if error := workflow._input_error():
            return workflow.result(False, error)
        workflow.executor.cancel_remaining()
        return workflow.cancel_plan(workflow.batch_id)
    if name == "confirm_plan":
        return await workflow.confirm(**arguments)
    if name == "update_plan_item":
        return await workflow.update(**arguments)
    if name == "remove_plan_item":
        return workflow.remove(**arguments)
    return workflow.cancel_plan(**arguments)
