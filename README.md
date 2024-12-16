# AI Researcher Agent

AI research assistant that performs comprehensive topic analysis and keyword research using a multi-agent system powered by CrewAI.

## Overview

This AI Researcher Agent employs a team of specialized agents working together to analyze research topics, identify optimal search terms, and compile structured recommendations. The system uses OpenAI's GPT-4 model and the Serper API for web searches.

## Key Features

- **Multi-Agent Architecture**: Utilizes three specialized agents:
  - Keyword Research Specialist
  - Search Results Analyst
  - Research Insights Compiler

- **Structured Output**: Generates three detailed reports:
  - Keyword analysis
  - Search result analysis
  - Final recommendations

## Requirements

- OpenAI API Key
- Serper API Key
- Python 3.8+

## Environment Variables

```env
OPENAI_API_KEY=your_openai_key
```
