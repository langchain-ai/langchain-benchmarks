from run_evals import main

experiments = [
    # {
    #     # "server_url": "http://localhost:1983/openai-functions-agent",
    #     "model": "openai-functions-agent",
    #     "project_name": "openai-functions-agent",
    # },
    {
        # "server_url": "http://localhost:1983/anthropic_chat",
        "model": "anthropic-chat",
        "project_name": "anthropic-chat",
    },
    # {
    #     "model": "chat",
    #     # "server_url": "http://localhost:1983/chat",
    #     "project_name": "chat",
    # },
    # Not worth our time it's so bad and slow
    # {
    #     # "server_url": "http://localhost:1983/anthropic_iterative_search",
    #     "model": "anthropic-iterative-search",
    #     "max_concurrency": 2,
    #     "project_name": "anthropic-iterative-search",
    # },
]

for experiment in experiments:
    main(**experiment, dataset_name="Chat Langchain Pub")
