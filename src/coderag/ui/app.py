"""Gradio web interface for CodeRAG."""

from typing import Optional

import gradio as gr

from coderag.ui.handlers import UIHandlers


def create_gradio_app() -> gr.Blocks:
    """Create the Gradio application."""
    handlers = UIHandlers()

    with gr.Blocks(title="CodeRAG - Code Q&A with Citations") as app:
        gr.Markdown("# CodeRAG - Code Q&A with Citations")
        gr.Markdown("Index GitHub repositories and ask questions about the code with verifiable citations.")

        with gr.Tabs():
            # Tab 1: Index Repository
            with gr.TabItem("Index Repository"):
                with gr.Row():
                    with gr.Column(scale=2):
                        repo_url = gr.Textbox(
                            label="GitHub Repository URL",
                            placeholder="https://github.com/owner/repo",
                            info="Enter a public GitHub repository URL",
                        )

                        with gr.Accordion("Advanced Options", open=False):
                            branch = gr.Textbox(
                                label="Branch",
                                placeholder="main",
                                value="",
                                info="Leave empty for default branch",
                            )
                            include_patterns = gr.Textbox(
                                label="Include Patterns",
                                placeholder="*.py, *.js, *.md",
                                value="",
                                info="Comma-separated glob patterns (leave empty for defaults)",
                            )
                            exclude_patterns = gr.Textbox(
                                label="Exclude Patterns",
                                placeholder="**/tests/**, **/node_modules/**",
                                value="",
                                info="Comma-separated glob patterns (leave empty for defaults)",
                            )

                        index_btn = gr.Button("Index Repository", variant="primary")

                    with gr.Column(scale=1):
                        index_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=3,
                        )
                        index_progress = gr.Progress()

                index_btn.click(
                    fn=handlers.index_repository,
                    inputs=[repo_url, branch, include_patterns, exclude_patterns],
                    outputs=[index_status],
                )

            # Tab 2: Ask Questions
            with gr.TabItem("Ask Questions"):
                with gr.Row():
                    with gr.Column(scale=2):
                        repo_selector = gr.Dropdown(
                            label="Select Repository",
                            choices=[],
                            interactive=True,
                        )
                        refresh_repos_btn = gr.Button("Refresh", size="sm")

                        question = gr.Textbox(
                            label="Question",
                            placeholder="Where is the function X defined?",
                            lines=2,
                        )

                        with gr.Row():
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Number of chunks to retrieve",
                            )

                        ask_btn = gr.Button("Ask", variant="primary")

                    with gr.Column(scale=1):
                        qa_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1,
                        )

                with gr.Row():
                    answer_output = gr.Markdown(label="Answer")

                with gr.Accordion("Evidence", open=True):
                    evidence_output = gr.Markdown(label="Retrieved Chunks")

                refresh_repos_btn.click(
                    fn=handlers.get_repositories,
                    outputs=[repo_selector],
                )

                ask_btn.click(
                    fn=handlers.ask_question,
                    inputs=[repo_selector, question, top_k],
                    outputs=[answer_output, evidence_output, qa_status],
                )

            # Tab 3: Manage Repositories
            with gr.TabItem("Manage Repositories"):
                repos_table = gr.Dataframe(
                    headers=["ID", "Repository", "Branch", "Chunks", "Status", "Indexed At"],
                    label="Indexed Repositories",
                    interactive=False,
                )

                with gr.Row():
                    refresh_table_btn = gr.Button("Refresh", size="sm")

                gr.Markdown("### Actions")

                with gr.Row():
                    with gr.Column(scale=2):
                        action_repo_id = gr.Textbox(
                            label="Repository ID",
                            placeholder="Enter repository ID (or first 8 characters)",
                            info="Copy the ID from the table above",
                        )
                    with gr.Column(scale=1):
                        update_btn = gr.Button("Update (Incremental)", variant="secondary")
                        delete_btn = gr.Button("Delete", variant="stop")

                action_status = gr.Textbox(label="Status", interactive=False, lines=5)

                refresh_table_btn.click(
                    fn=handlers.get_repositories_table,
                    outputs=[repos_table],
                )

                update_btn.click(
                    fn=handlers.index_repository_incremental,
                    inputs=[action_repo_id],
                    outputs=[action_status],
                )

                delete_btn.click(
                    fn=handlers.delete_repository,
                    inputs=[action_repo_id],
                    outputs=[action_status, repos_table],
                )

        # Load initial data
        app.load(
            fn=handlers.get_repositories,
            outputs=[repo_selector],
        )

    return app
