import os
import sys
import asyncio
from typing import Any

from github import Github
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import (
    FunctionAgent, AgentWorkflow, AgentOutput, ToolCall, ToolCallResult
)
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

# --- Read args passed in from the GitHub Actions workflow step ---
# run: poetry run python agent.py $GITHUB_TOKEN $REPOSITORY $PR_NUMBER $OPENAI_API_KEY $OPENAI_BASE_URL
github_token = sys.argv[1]
full_repo_name = sys.argv[2]          # e.g. "username/recipes-api"
pr_number = int(sys.argv[3])
openai_api_key = sys.argv[4]
openai_base_url = sys.argv[5] if len(sys.argv) > 5 else None

git = Github(github_token)
repo = git.get_repo(full_repo_name)

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
    api_base=openai_base_url,
)


def get_pr_details(pr_number: int) -> dict[str, Any]:
    """Use this tool to get details about a pull request given its number.
    Returns author, title, body, diff_url, state, and all commit SHAs."""
    pr = repo.get_pull(pr_number)

    commit_shas = []
    for c in pr.get_commits():
        commit_shas.append(c.sha)

    return {
        "author": pr.user.login,
        "title": pr.title,
        "body": pr.body,
        "diff_url": pr.diff_url,
        "state": pr.state,
        "commit_shas": commit_shas,
        "head_sha": commit_shas[-1] if commit_shas else None,
    }


def get_file_content(file_path: str) -> str:
    """Use this tool to fetch the contents of a specific file from the repository,
    given its file path."""
    return repo.get_contents(file_path).decoded_content.decode('utf-8')


def get_commit_details(head_sha: str) -> list[dict[str, Any]]:
    """Use this tool to get details about a specific commit given its SHA.
    Returns the list of changed files with their status, additions,
    deletions, changes and diff patch."""
    commit = repo.get_commit(head_sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


def post_review_to_github(pr_number: int, comment: str) -> str:
    pr = repo.get_pull(pr_number)
    pr.create_review(body=comment, event="COMMENT")
    return f"Review posted successfully on PR #{pr_number}."


async def add_context_to_state(ctx: Context, gathered_context: str) -> str:
    """Useful for adding the gathered PR context to the state."""
    current_state = await ctx.store.get("state")
    current_state["gathered_contexts"] = gathered_context
    await ctx.store.set("state", current_state)
    return "State updated with gathered context."


async def add_comment_to_state(ctx: Context, draft_comment: str) -> str:
    """Useful for adding the draft PR review comment to the state."""
    current_state = await ctx.store.get("state")
    current_state["draft_comment"] = draft_comment
    await ctx.store.set("state", current_state)
    return "State updated with draft comment."


async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final, approved PR review to the state."""
    current_state = await ctx.store.get("state")
    current_state["final_review"] = final_review
    await ctx.store.set("state", current_state)
    return "State updated with final review."


pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_content_tool = FunctionTool.from_defaults(get_file_content)
commit_details_tool = FunctionTool.from_defaults(get_commit_details)
post_review_to_github_tool = FunctionTool.from_defaults(post_review_to_github)
add_context_to_state_tool = FunctionTool.from_defaults(add_context_to_state)
add_comment_to_state_tool = FunctionTool.from_defaults(add_comment_to_state)
add_final_review_to_state_tool = FunctionTool.from_defaults(add_final_review_to_state)

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from a GitHub pull request, "
                 "including PR details, changed files, and any requested repo files.",
    tools=[pr_details_tool, file_content_tool, commit_details_tool, add_context_to_state_tool],
    system_prompt=(
        "You are the context gathering agent. When gathering context, you MUST gather \n"
        "  - The details: author, title, body, diff_url, state, and head_sha; \n"
        "  - Changed files; \n"
        "  - Any requested for files; \n"
        "Once you gather the requested info, you MUST hand control back to the Commentor Agent. "
    ),
    can_handoff_to=["CommentorAgent"],
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[add_comment_to_state_tool],
    system_prompt=(
        "You are the commentor agent that writes review comments for pull requests as a human reviewer would.\n"
        "Your task has TWO phases, and you are not finished until BOTH are complete:\n\n"
        "PHASE 1 - Gather context and draft the review:\n"
        " - Request the PR details, changed files, and any other repo files you may need from the ContextAgent.\n"
        " - Once you have all needed information, draft a ~200-300 word review in markdown format detailing:\n"
        "    - What is good about the PR?\n"
        "    - Did the author follow ALL contribution rules? What is missing?\n"
        "    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this.\n"
        "    - Are new endpoints documented? - use the diff to determine this.\n"
        "    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement.\n"
        " - Address the author directly, as a human reviewer would.\n"
        " - If you need any additional details, hand off to the ContextAgent, then resume Phase 1 once you have it.\n\n"
        "PHASE 2 - MANDATORY final steps (never skip, never respond to the user before doing these):\n"
        " 1. Call the add_comment_to_state tool, passing your full drafted review as the draft_comment argument.\n"
        " 2. Immediately call the handoff tool to hand off to ReviewAndPostingAgent.\n\n"
        "You must NEVER produce a final text-only answer to the user. Writing the review text is only "
        "an intermediate step - your turn is not over until you have called add_comment_to_state AND "
        "handed off to ReviewAndPostingAgent, in that exact order. If you find yourself about to give a "
        "final answer without having called both tools, stop and call them first."
    ),
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"],
)


review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the drafted PR comment for completeness and quality, "
                 "requests rewrites from the CommentorAgent if needed, and posts "
                 "the final approved review to GitHub.",
    tools=[add_final_review_to_state_tool, post_review_to_github_tool],
    system_prompt=(
        "You are the Review and Posting agent, responsible for the final quality check and posting the review.\n\n"
        "STEP 1: If no draft review exists yet in the state, hand off to CommentorAgent to produce one.\n\n"
        "STEP 2: Once a draft_comment exists in state, evaluate it against this checklist:\n"
        "   - Is it a ~200-300 word review in markdown format?\n"
        "   - Does it specify what is good about the PR?\n"
        "   - Does it note whether the author followed ALL contribution rules, and what's missing?\n"
        "   - Does it note test availability for new functionality, and migrations for new models?\n"
        "   - Does it note whether new endpoints were documented?\n"
        "   - Does it suggest specific line improvements with quotes?\n"
        " If the review fails ANY of these checks, hand off to CommentorAgent with specific feedback on what to fix, "
        "and repeat Step 2 once you receive the revised draft.\n\n"
        "STEP 3: MANDATORY - once the review passes the checklist, you must, without exception:\n"
        "   1. Call add_final_review_to_state with the approved review text.\n"
        "   2. Call post_review_to_github with the PR number and the approved review text as the comment.\n\n"
        "You must NEVER end your turn or give a final answer without having called post_review_to_github. "
        "Posting the review to GitHub is the only successful completion of your task."
    ),
    can_handoff_to=["CommentorAgent"],
)


workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "draft_comment": "",
        "final_review": "",
    },
)


async def main():
    query = f"Write a review for PR number {pr_number}."

    handler = workflow_agent.run(query)

    current_agent = None
    final_response_text = None
    posted_to_github = False

    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                final_response_text = event.response.content
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
            if event.tool_name == "post_review_to_github":
                posted_to_github = True
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")

    result = await handler
    print("\n\nWorkflow finished. Result:", result)

    # Safety net: if the agents never actually posted the review, do it ourselves
    if not posted_to_github and final_response_text:
        print("\n\n[Fallback] Review was drafted but never posted. Posting it now directly...")
        outcome = post_review_to_github(pr_number, final_response_text)
        print(outcome)
    elif posted_to_github:
        print("\n\nReview was successfully posted by the agent workflow.")
    else:
        print("\n\n[Warning] No final review text was captured. Nothing to post.")



if __name__ == "__main__":
    asyncio.run(main())
    git.close()
