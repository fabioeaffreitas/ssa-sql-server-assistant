\# SSA — SQL Server AI Assistant



\*\*A local AI-powered assistant for SQL Server using Retrieval-Augmented Generation (RAG) and Text-to-SQL.\*\*



\## Overview



SSA (SQL Server AI Assistant) is a personal engineering project exploring how Large Language Models (LLMs) can interact with relational databases through natural language.



The objective is to build a local AI assistant capable of interpreting user questions, retrieving database schema context, generating SQL queries, validating them, and executing authorized read-only operations against Microsoft SQL Server.



\## Project Objectives



\- Enable natural-language interaction with SQL Server databases.

\- Implement Retrieval-Augmented Generation (RAG) for database schema awareness.

\- Generate T-SQL using a locally hosted LLM.

\- Introduce SQL validation and execution safeguards.

\- Keep inference and database access within a local development environment.

\- Demonstrate practical applications of Generative AI in data engineering.



\## Technology Stack



| Technology | Purpose |

|---|---|

| Python | Application development |

| Microsoft SQL Server 2025 | Relational database |

| AdventureWorks2025 | Development and testing database |

| Ollama | Local LLM runtime |

| Qwen2.5-Coder | Local language model |

| ChromaDB | Planned vector database |

| RAG | Schema retrieval and contextual grounding |

| Git / GitHub | Version control and project documentation |



\## Architecture



Planned high-level workflow:



1\. User submits a natural-language question.

2\. SSA retrieves relevant database schema context.

3\. The LLM generates a T-SQL query.

4\. SSA validates the generated SQL.

5\. The application executes authorized read-only queries.

6\. Results are returned in a user-friendly format.



\## Security Principles



\- Read-only database access for query execution.

\- SQL validation before execution.

\- Explicit separation between query generation and execution.

\- Local LLM inference through Ollama.

\- No unrestricted database operations.



\## Development Status



\*\*Status: Initial development\*\*



| Component | Status |

|---|---|

| Local development environment | Completed |

| SQL Server connectivity | Completed |

| Ollama integration | Verified manually |

| Schema retrieval | Planned |

| RAG implementation | Planned |

| Text-to-SQL pipeline | Planned |

| SQL validation | Planned |

| Read-only execution layer | Planned |

| User interface | Planned |



\## Roadmap



\- \[x] Development environment setup

\- \[x] SQL Server connectivity

\- \[x] Local LLM installation and testing

\- \[x] GitHub repository initialization

\- \[ ] Database schema introspection

\- \[ ] RAG-based schema retrieval

\- \[ ] Text-to-SQL generation pipeline

\- \[ ] SQL validation and safety controls

\- \[ ] Read-only query execution

\- \[ ] Result presentation

\- \[ ] Demonstration and portfolio documentation



\## Disclaimer



SSA is an experimental engineering project intended for learning, research, and demonstration purposes. Generated SQL must be validated before execution, and database permissions remain the primary execution safeguard.



\---



\*\*Built as a practical exploration of Generative AI, SQL Server, and data engineering.\*\*



\## Testing



\### Deterministic regression tests



Run the regression suite without requiring Ollama:



```powershell

python -m pytest tests/test\_sql\_patterns.py tests/test\_sql\_validator.py tests/test\_sql\_schema\_validator.py tests/test\_sql\_executor.py tests/test\_assistant.py -v

```



\### SQL generation evaluation



Run live SQL generation tests using the configured local Ollama model:



```powershell

python -m pytest tests/test\_sql\_generation\_eval.py -v

```



The evaluation tests call the language model and may take several minutes. Results can vary between runs because SQL generation is probabilistic.





