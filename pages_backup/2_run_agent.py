# pages/2_run_agent.py
# Streamlit page: Run the LangGraph agent and stream node outputs.
#
# TODO:
#   - Load JD and resumes from session state (set by upload page)
#   - Invoke recruitment_graph with initial AgentState
#   - Stream each node result to the UI as it completes
#   - Display iteration counter and current node name
#   - Halt at human_approval_node for user action (approve / reject)
