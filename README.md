# Nexus Intelligence - Multi-Agent Business Intelligence System

A sophisticated multi-agent system that automates comprehensive business intelligence analysis using LangChain, LangGraph, and multiple data sources. The system orchestrates 9 specialized agents to deliver actionable market insights, competitive analysis, and strategic recommendations.

## 🏗️ System Architecture

IntelliGraph uses a sequential multi-agent workflow powered by LangGraph, where each agent specializes in a specific aspect of business intelligence:

```
Controller → MarketWatcher → CompetitorMonitor → FundingIntel → TrendAnalysis 
    ↓
StrategyAdvisor → ExecutionPlanner → ResourceRecommender → RiskAnalyzer → ReportAgent
```

### Agent Responsibilities

1. **MarketWatcherAgent**: Discovers competitor websites and recent news using Serper API
2. **CompetitorMonitorAgent**: Scrapes competitor websites for product and positioning analysis
3. **FundingIntelAgent**: Monitors funding events and startup news via TechCrunch RSS
4. **TrendAnalysisAgent**: Analyzes Google Trends data for market interest patterns
5. **StrategyAdvisorAgent**: Synthesizes data into strategic insights and opportunities
6. **ExecutionPlannerAgent**: Creates actionable execution plans with timelines
7. **ResourceRecommenderAgent**: Suggests specific tools, platforms, and resources
8. **RiskAnalyzerAgent**: Identifies risks and develops mitigation strategies
9. **ReportAgent**: Generates comprehensive markdown reports

## 🚀 Features

- **Multi-Agent Coordination**: Sequential workflow with state management
- **Persistent Memory**: JSON-based memory system for agent data sharing
- **Multiple Data Sources**: Integrates 4+ external APIs and RSS feeds
- **FastAPI Integration**: RESTful API with authentication and file downloads
- **Comprehensive Reports**: Professional markdown reports with strategic insights
- **Error Handling**: Robust error management and API key validation
- **Rate Limiting**: Built-in delays to respect API rate limits

## 📋 Prerequisites

### Required API Keys

Create a `.env` file in the project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
SERPER_API_KEY=your_serper_api_key_here
SCRAPER_API_KEY=your_scraper_api_key_here
SERP_API_KEY=your_serp_api_key_here
```

### API Key Sources

- **Groq API**: [console.groq.com](https://console.groq.com) - For LLM processing
- **Serper API**: [serper.dev](https://serper.dev) - For Google search and news
- **ScraperAPI**: [scraperapi.com](https://scraperapi.com) - For website scraping
- **SerpAPI**: [serpapi.com](https://serpapi.com) - For Google Trends data

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd intelligraph
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env file with your API keys
```

4. Run the API server:
```bash
python main.py
```

The API will be available at : https://nexus-intelligence1.onrender.com

## 🌐 Live Demo

**Deployed API**: https://nexus-intelligence1.onrender.com

- **Interactive Documentation**: https://nexus-intelligence1.onrender.com/docs
- **Health Check**: https://nexus-intelligence1.onrender.com/health

## 📖 API Usage

### Authentication

All endpoints require an API key in the header:
```bash
X-API-Key: apikey
```

### Endpoints

#### 1. Get Analysis as JSON
```bash
POST /get_file_json
Content-Type: application/json
X-API-Key: mysecret123

{
    "task": "Analyze the AI/ML market for investment opportunities",
    "competitors": ["OpenAI", "Anthropic", "Cohere", "Hugging Face"]
}
```

#### 2. Download Report File
```bash
POST /download_file
Content-Type: application/json
X-API-Key: api_key

{
    "task": "Evaluate electric vehicle market positioning",
    "competitors": ["Tesla", "Rivian", "Lucid Motors"]
}
```

#### 3. Health Check
```bash
GET /health
```

### Example Request

```python
import requests

url = "http://localhost:8000/get_file_json"
headers = {
    "X-API-Key": "api_key",
    "Content-Type": "application/json"
}
data = {
    "task": "Analyze fintech market for new product launch",
    "competitors": ["Stripe", "Square", "PayPal", "Adyen"]
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result["content"])  # Full analysis report
```

## 📊 Output Format

The system generates comprehensive reports including:

- **Executive Summary**: High-level findings and recommendations
- **Market Intelligence**: Competitor websites, news, and market positioning
- **Competitive Analysis**: Website content analysis and positioning insights
- **Funding Intelligence**: Recent funding events and investor interest
- **Trend Analysis**: Google Trends data and search patterns
- **Strategic Recommendations**: Data-driven strategic opportunities
- **Execution Plan**: Actionable steps with timelines and resources
- **Resource Requirements**: Specific tools and platforms recommended
- **Risk Analysis**: Potential risks and mitigation strategies

## 🔧 Project Structure

```
intelligraph/
├── main.py                          # FastAPI application
├── business_intelligence.py         # Multi-agent system core
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (create this)
├── project_memory1.json           # Agent memory storage (auto-created)
├── business_intelligence_report.md # Generated reports (auto-created)
└── README.md                       # This file
```

## 🚦 Usage Examples

### Quick Start
```python
from business_intelligence import run_pipeline

# Run analysis with custom parameters
report_file = run_pipeline(
    task="Analyze SaaS market for expansion opportunities",
    competitors=["Salesforce", "HubSpot", "Zendesk"]
)
print(f"Report saved to: {report_file}")
```

### Memory Management
```python
from business_intelligence import view_memory, clear_memory, check_api_keys

# Check API configuration
check_api_keys()

# View current memory contents
view_memory()

# Clear memory for fresh analysis
clear_memory()
```

## ⚙️ Configuration

### API Rate Limits

The system includes built-in rate limiting:
- **Serper API**: 1-second delays between requests
- **ScraperAPI**: 2-second delays between websites
- **SerpAPI**: 5-second delays between trend queries
- **TrendAnalysis**: Limited to 2 competitors to minimize API usage

### Memory System

- **Persistent Storage**: Agent data saved to `project_memory1.json`
- **Cross-Agent Communication**: Agents access previous agent results
- **Error Recovery**: System continues if individual agents fail

## 🛡️ Error Handling

The system includes comprehensive error handling:

- **API Key Validation**: Checks for missing keys before execution
- **Request Timeouts**: Prevents hanging on slow API responses
- **Graceful Degradation**: Continues analysis even if some data sources fail
- **Memory Persistence**: Saves progress to recover from interruptions

## 🔍 Monitoring and Debugging

### Console Output
The system provides detailed logging:
```
MarketWatcherAgent starting...
Analyzing competitors: ['OpenAI', 'Anthropic']
Processing OpenAI...
Found website for OpenAI: https://openai.com
Controller Decision: CompetitorMonitorAgent
```

### Memory Inspection
```python
# View what data each agent collected
view_memory()
```

### Health Monitoring
```bash
curl -X GET "http://localhost:8000/health"
```

## 📈 Performance Considerations

- **Sequential Processing**: Agents run in sequence to ensure data dependencies
- **Memory Efficient**: Uses JSON for lightweight data storage
- **API Optimization**: Minimal requests with strategic delays
- **Timeout Management**: 10-30 second timeouts prevent hanging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-agent`
3. Add your agent following the existing pattern
4. Update the controller logic to include your agent
5. Test with sample data
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🔗 API Documentation

Once the server is running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🆘 Support

For issues and questions:
1. Check the console output for detailed error messages
2. Verify API key configuration with `check_api_keys()`
3. Review memory contents with `view_memory()`
4. Check API rate limits if experiencing timeouts

## 🔮 Future Enhancements

- **Parallel Processing**: Run compatible agents in parallel
- **Database Integration**: Replace JSON memory with database
- **Real-time Updates**: WebSocket support for live analysis
- **Custom Agents**: Plugin system for user-defined agents
- **Advanced Analytics**: Machine learning insights and predictions
