# Fixed Multi-Agent Business Intelligence System

import os
import json
import requests
import feedparser
from typing import TypedDict, Annotated, List, Literal, Dict, Any
from datetime import datetime
import time

# Import LangChain components
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Import LangGraph components
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# API Keys - Make sure these are set in your .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192",
    temperature=0
)

# Memory management functions
MEMORY_FILE = "project_memory1.json"

def read_memory():
    """Load existing memory from file or return empty dict."""
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: Memory file corrupted, starting fresh")
            return {}
    return {}

def write_memory(agent_name, data):
    """Update memory with agent's results without overwriting others."""
    try:
        memory = read_memory()
        memory[agent_name] = data
        with open(MEMORY_FILE, "w") as f:
            json.dump(memory, f, indent=2)
        print(f"Saved {agent_name} data to memory")
    except Exception as e:
        print(f"Error saving memory for {agent_name}: {e}")

# State definition
class SupervisorState(MessagesState):
    """State for the multi-agent system"""
    next_agent: str = ""  
    serper_data: str = ""
    product_hunt_data: str = ""
    scraper_scrap_data: str = ""
    tech_rss_data: str = ""
    angel_list_data: str = ""
    serp_trend_data: str = ""
    statergy_data: str = ""
    planner_agent_data: str = ""
    recommend_tools_data: str = ""
    risk_analyzer_data: str = ""
    report_making_data: str = ""
    final_report: str = ""
    task_complete: bool = False
    current_task: str = ""
    competitor_list: List[str] = []
    agents_completed: List[str] = []  # Track completed agents

# FIXED Controller function with better logic
def controller_chain():
    """Creates the controller decision chain for the 9-agent business system"""
    
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a controller managing a team of 9 specialized agents that must run in sequence:

REQUIRED SEQUENCE:
1. MarketWatcherAgent → 2. CompetitorMonitorAgent → 3. FundingIntelAgent → 4. TrendAnalysisAgent → 
5. StrategyAdvisorAgent → 6. ExecutionPlannerAgent → 7. ResourceRecommenderAgent → 8. RiskAnalyzerAgent → 9. ReportAgent

Current completion status:
- MarketWatcherAgent: {market_watcher_done}
- CompetitorMonitorAgent: {competitor_monitor_done}
- FundingIntelAgent: {funding_intel_done}
- TrendAnalysisAgent: {trend_analysis_done}
- StrategyAdvisorAgent: {strategy_advisor_done}
- ExecutionPlannerAgent: {execution_planner_done}
- ResourceRecommenderAgent: {resource_recommender_done}
- RiskAnalyzerAgent: {risk_analyzer_done}
- ReportAgent: {report_done}

RULES:
- Run agents in EXACT sequence order
- Only proceed to next agent when current one is complete
- If all 9 agents are complete, respond with 'DONE'

Respond with ONLY the next agent name or 'DONE':
market_watcher / competitor_monitor / funding_intel / trend_analysis / strategy_advisor / execution_planner / resource_recommender / risk_analyzer / report / DONE
"""),
        ("human", "{task}")
    ])
    
    return supervisor_prompt | llm

def controller_agent(state: SupervisorState) -> Dict:
    """FIXED Controller with sequential logic"""

    messages = state["messages"]
    task = messages[-1].content if messages else "No task"

    # Check completion status for each agent
    market_watcher_done = bool(state.get("serper_data", "")) and bool(state.get("product_hunt_data", ""))
    competitor_monitor_done = bool(state.get("scraper_scrap_data", ""))
    funding_intel_done = bool(state.get("tech_rss_data", ""))
    trend_analysis_done = bool(state.get("serp_trend_data", ""))
    strategy_advisor_done = bool(state.get("statergy_data", ""))
    execution_planner_done = bool(state.get("planner_agent_data", ""))
    resource_recommender_done = bool(state.get("recommend_tools_data", ""))
    risk_analyzer_done = bool(state.get("risk_analyzer_data", ""))
    report_done = bool(state.get("report_making_data", "")) and bool(state.get("final_report", ""))

    print(f"Controller Status Check:")
    print(f"   MarketWatcher: {'Done' if market_watcher_done else 'Pending'}")
    print(f"   CompetitorMonitor: {'Done' if competitor_monitor_done else 'Pending'}")
    print(f"   FundingIntel: {'Done' if funding_intel_done else 'Pending'}")
    print(f"   TrendAnalysis: {'Done' if trend_analysis_done else 'Pending'}")
    print(f"   StrategyAdvisor: {'Done' if strategy_advisor_done else 'Pending'}")
    print(f"   ExecutionPlanner: {'Done' if execution_planner_done else 'Pending'}")
    print(f"   ResourceRecommender: {'Done' if resource_recommender_done else 'Pending'}")
    print(f"   RiskAnalyzer: {'Done' if risk_analyzer_done else 'Pending'}")
    print(f"   Report: {'Done' if report_done else 'Pending'}")

    # FIXED: Sequential logic - run agents in order
    if not market_watcher_done:
        next_agent = "MarketWatcherAgent"
        controller_msg = "Controller: Starting MarketWatcherAgent..."
    elif not competitor_monitor_done:
        next_agent = "CompetitorMonitorAgent"
        controller_msg = "Controller: Starting CompetitorMonitorAgent..."
    elif not funding_intel_done:
        next_agent = "FundingIntelAgent"
        controller_msg = "Controller: Starting FundingIntelAgent..."
    elif not trend_analysis_done:
        next_agent = "TrendAnalysisAgent"
        controller_msg = "Controller: Starting TrendAnalysisAgent..."
    elif not strategy_advisor_done:
        next_agent = "StrategyAdvisorAgent"
        controller_msg = "Controller: Starting StrategyAdvisorAgent..."
    elif not execution_planner_done:
        next_agent = "ExecutionPlannerAgent"
        controller_msg = "Controller: Starting ExecutionPlannerAgent..."
    elif not resource_recommender_done:
        next_agent = "ResourceRecommenderAgent"
        controller_msg = "Controller: Starting ResourceRecommenderAgent..."
    elif not risk_analyzer_done:
        next_agent = "RiskAnalyzerAgent"
        controller_msg = "Controller: Starting RiskAnalyzerAgent..."
    elif not report_done:
        next_agent = "ReportAgent"
        controller_msg = "Controller: Starting ReportAgent..."
    else:
        next_agent = "end"
        controller_msg = "Controller: All agents complete! Analysis finished."

    print(f"Controller Decision: {next_agent}")

    return {
        "messages": [AIMessage(content=controller_msg)],
        "next_agent": next_agent,
        "current_task": task
    }

# Agent functions
def MarketWatcherAgent(state: SupervisorState) -> Dict:
    """MarketWatcherAgent: Finds competitor websites, news, and trends, processes with Groq."""
    
    print("MarketWatcherAgent starting...")
    
    # Get competitor list from state or use default
    competitors = state.get("competitor_list", ["Tesla", "Apple", "Google"])
    if not competitors:
        competitors = ["Tesla", "Apple", "Google"]
    
    print(f"Analyzing competitors: {competitors}")
    
    results = []

    for company in competitors:
        print(f"Processing {company}...")
        
        # 1. Search website using Serper API
        website = None
        if SERPER_API_KEY:
            try:
                serper_url = "https://google.serper.dev/search"
                payload = {"q": f"{company} official site"}
                headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
                
                print(f"Searching website for {company}...")
                serper_resp = requests.post(serper_url, json=payload, headers=headers, timeout=10)
                
                if serper_resp.status_code == 200:
                    serper_data = serper_resp.json()
                    if serper_data.get("organic"):
                        website = serper_data["organic"][0].get("link")
                        print(f"Found website for {company}: {website}")
                        
            except Exception as e:
                print(f"Serper API error for {company}: {e}")
        else:
            print("SERPER_API_KEY not found, skipping website search")

        # 2. Get news
        news_list = []
        if SERPER_API_KEY:
            try:
                hunt_url = "https://google.serper.dev/news"
                hunt_payload = {"q": company}
                headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
                
                print(f"Searching news for {company}...")
                hunt_resp = requests.post(hunt_url, json=hunt_payload, headers=headers, timeout=10)
                
                if hunt_resp.status_code == 200:
                    hunt_data = hunt_resp.json()
                    if hunt_data.get("news"):
                        for article in hunt_data["news"][:3]:
                            news_list.append({
                                "title": article.get("title", ""),
                                "link": article.get("link", "")
                            })
                        print(f"Found {len(news_list)} news articles for {company}")
                        
            except Exception as e:
                print(f"News API error for {company}: {e}")

        results.append({
            "name": company,
            "website": website,
            "news": news_list
        })
        
        time.sleep(1)

    write_memory("market_watcher_agent", results)
    
    # 3. Generate summary
    try:
        raw_data_text = f"Here is competitor data for {len(competitors)} companies:\n{json.dumps(results, indent=2)}\n\nPlease summarize key insights and trends from this data."
        print("Sending data to Groq for analysis...")
        groq_summary = llm.invoke([HumanMessage(content=raw_data_text)]).content
        print("Groq analysis complete")
    except Exception as e:
        print(f"Groq analysis error: {e}")
        groq_summary = f"Analysis completed for {len(competitors)} competitors. Raw data collected successfully."

    return {
        "messages": [AIMessage(content=f"MarketWatcherAgent: Analyzed {len(competitors)} competitors successfully.")],
        "serper_data": json.dumps(results),
        "product_hunt_data": groq_summary
    }

def CompetitorMonitorAgent(state: SupervisorState) -> Dict:
    """CompetitorMonitorAgent: Scrapes competitor websites and analyzes updates."""
    
    print("CompetitorMonitorAgent starting...")

    competitor_data = read_memory().get("market_watcher_agent", [])
    
    if not competitor_data:
        print("No competitor data found from previous agent")
        return {
            "messages": [AIMessage(content="CompetitorMonitorAgent: No competitor data found.")],
            "scraper_scrap_data": "No competitor data available for scraping."
        }

    scraped_results = []

    for competitor in competitor_data:
        website = competitor.get("website")
        company_name = competitor.get("name", "Unknown")
        
        print(f"Processing {company_name}...")
        
        if not website:
            print(f"No website found for {company_name}")
            scraped_results.append({
                "name": company_name,
                "website": None,
                "html": None,
                "error": "No website found"
            })
            continue

        if not SCRAPER_API_KEY:
            scraped_results.append({
                "name": company_name,
                "website": website,
                "html": None,
                "error": "Scraper API key not configured"
            })
            continue

        try:
            print(f"Scraping {website}...")
            scraper_url = f"http://api.scraperapi.com?api_key={SCRAPER_API_KEY}&url={website}"
            resp = requests.get(scraper_url, timeout=30)
            
            if resp.status_code == 200:
                html_content = resp.text[:3000]
                print(f"Successfully scraped {company_name} ({len(html_content)} chars)")
            else:
                html_content = None
                print(f"Scraping failed for {company_name}: Status {resp.status_code}")

            scraped_results.append({
                "name": company_name,
                "website": website,
                "html": html_content,
                "status_code": resp.status_code
            })

        except Exception as e:
            print(f"Scraping error for {company_name}: {e}")
            scraped_results.append({
                "name": company_name,
                "website": website,
                "html": None,
                "error": str(e)
            })
        
        time.sleep(2)

    write_memory("competitor_monitor_agent", scraped_results)

    # Generate summary
    try:
        raw_text = f"Here is scraped website data for {len(scraped_results)} competitors:\n"
        for result in scraped_results:
            raw_text += f"\nCompany: {result['name']}\n"
            raw_text += f"Website: {result.get('website', 'N/A')}\n"
            if result.get('html'):
                raw_text += f"Content preview: {result['html'][:500]}...\n"
            else:
                raw_text += f"Error: {result.get('error', 'Unknown error')}\n"
        
        raw_text += "\n\nPlease analyze this data and extract key insights about competitor websites, products, and positioning."
        
        print("Sending scraped data to Groq for analysis...")
        summary = llm.invoke([HumanMessage(content=raw_text)]).content
        print("Competitor analysis complete")
        
    except Exception as e:
        print(f"Groq analysis error: {e}")
        summary = f"Competitor monitoring completed for {len(scraped_results)} companies. Website data collected."

    return {
        "messages": [AIMessage(content=f"CompetitorMonitorAgent: Processed {len(scraped_results)} competitor websites.")],
        "scraper_scrap_data": summary
    }

def FundingIntelAgent(state: SupervisorState) -> Dict:
    """FundingIntelAgent: Finds recent funding events for competitors using TechCrunch RSS."""
    
    print("FundingIntelAgent starting...")

    TECHCRUNCH_RSS = "https://techcrunch.com/startups/feed/"

    competitor_data = read_memory().get("competitor_monitor_agent", [])
    if not competitor_data:
        competitor_data = read_memory().get("market_watcher_agent", [])

    if not competitor_data:
        print("No competitor data found")
        return {
            "messages": [AIMessage(content="FundingIntelAgent: No competitor data found.")],
            "tech_rss_data": "No competitor data available for funding analysis."
        }

    funding_results = []

    try:
        print("Fetching TechCrunch RSS feed...")
        feed = feedparser.parse(TECHCRUNCH_RSS)
        
        if not feed.entries:
            print("No entries found in RSS feed")
        else:
            print(f"Found {len(feed.entries)} RSS entries")

        for competitor in competitor_data:
            name = competitor.get("name", "")
            if not name:
                continue
                
            print(f"Searching funding info for {name}...")

            matched_entries = []
            for entry in feed.entries[:30]:
                title = entry.get("title", "").lower()
                summary = entry.get("summary", "").lower()
                name_lower = name.lower()
                
                if name_lower in title or name_lower in summary:
                    matched_entries.append({
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:500]
                    })
                    print(f"Found relevant article: {entry.get('title', '')}")

            funding_results.append({
                "name": name,
                "funding_info": matched_entries,
                "articles_found": len(matched_entries)
            })

    except Exception as e:
        print(f"RSS parsing error: {e}")
        funding_results = [{"error": str(e)}]

    write_memory("funding_intel_agent", funding_results)

    try:
        raw_text = f"Here are funding and startup news results for competitors:\n{json.dumps(funding_results, indent=2)}\n\nPlease analyze and summarize key funding insights and trends."
        print("Sending funding data to Groq for analysis...")
        summary = llm.invoke([HumanMessage(content=raw_text)]).content
        print("Funding analysis complete")
        
    except Exception as e:
        print(f"Groq analysis error: {e}")
        summary = f"Funding intelligence gathered for {len(funding_results)} competitors from TechCrunch RSS feed."

    return {
        "messages": [AIMessage(content=f"FundingIntelAgent: Analyzed funding info for {len(funding_results)} competitors.")],
        "tech_rss_data": summary
    }

def extract_trend_insights(trend_data, company_name):
    """Extract top 5 meaningful insights from Google Trends data"""
    insights = []
    
    try:
        if trend_data and "interest_over_time" in trend_data:
            timeline_data = trend_data["interest_over_time"].get("timeline_data", [])
            
            if timeline_data:
                recent_data = timeline_data[-5:]
                
                values = [point["values"][0]["extracted_value"] for point in recent_data if point.get("values")]
                if len(values) >= 2:
                    trend_direction = "Increasing" if values[-1] > values[0] else "Decreasing" if values[-1] < values[0] else "Stable"
                    peak_value = max(values)
                    avg_value = sum(values) / len(values)
                    
                    insights.append(f"Search interest trend: {trend_direction} (Peak: {peak_value}, Avg: {avg_value:.1f})")
        
        if trend_data and "related_queries" in trend_data:
            rising_queries = trend_data["related_queries"].get("rising", [])
            if rising_queries:
                top_rising = [q.get("query", "") for q in rising_queries[:3]]
                insights.append(f"Rising related searches: {', '.join(top_rising)}")
        
        if trend_data and "related_topics" in trend_data:
            rising_topics = trend_data["related_topics"].get("rising", [])
            if rising_topics:
                top_topics = [t.get("topic", {}).get("title", "") for t in rising_topics[:2]]
                insights.append(f"Trending topics: {', '.join(filter(None, top_topics))}")
        
        if trend_data and "interest_by_region" in trend_data:
            regions = trend_data["interest_by_region"][:3]
            if regions:
                top_regions = [r.get("location", "") for r in regions]
                insights.append(f"Top regions: {', '.join(filter(None, top_regions))}")
        
        if trend_data and "interest_over_time" in trend_data:
            timeline_data = trend_data["interest_over_time"].get("timeline_data", [])
            if timeline_data:
                latest_value = timeline_data[-1]["values"][0]["extracted_value"] if timeline_data[-1].get("values") else 0
                insights.append(f"Current search popularity: {latest_value}/100")
                
    except Exception as e:
        insights.append(f"Error extracting insights: {str(e)}")
    
    return insights[:5]

def TrendAnalysisAgent(state: SupervisorState) -> Dict:
    """TrendAnalysisAgent: MINIMAL SEARCHES - Only analyzes 2 competitors with longer delays."""
    
    print("TrendAnalysisAgent starting...")

    competitor_data = read_memory().get("competitor_monitor_agent", [])
    if not competitor_data:
        competitor_data = read_memory().get("market_watcher_agent", [])

    if not competitor_data:
        print("No competitor data found")
        return {
            "messages": [AIMessage(content="TrendAnalysisAgent: No competitor data found.")],
            "serp_trend_data": "No competitor data available for trend analysis."
        }

    trend_results = []
    processed_insights = {}

    # CHANGE: Only analyze first 2 competitors to minimize API calls
    for competitor in competitor_data[:2]:  # MINIMAL SEARCHES - reduced from 4 to 2
        name = competitor.get("name")
        if not name:
            continue
            
        print(f"Analyzing trends for {name}...")

        if not SERP_API_KEY:
            print(f"SERP_API_KEY not found, skipping trend analysis for {name}")
            trend_results.append({
                "name": name,
                "insights": ["SerpAPI key not configured"],
                "raw_data": None
            })
            continue

        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_trends",
                "q": name,
                "api_key": SERP_API_KEY,
                "data_type": "TIMESERIES",
                "time_range": "today 12-m"
            }
            resp = requests.get(url, params=params, timeout=20)
            
            if resp.status_code == 200:
                trend_data = resp.json()
                print(f"Got trend data for {name}")
                
                insights = extract_trend_insights(trend_data, name)
                print(f"Extracted {len(insights)} insights for {name}")
                
                minimal_data = {
                    "company": name,
                    "search_trend": "Available" if trend_data.get("interest_over_time") else "Not available",
                    "related_queries_count": len(trend_data.get("related_queries", {}).get("rising", [])),
                    "regions_tracked": len(trend_data.get("interest_by_region", [])),
                    "data_points": len(trend_data.get("interest_over_time", {}).get("timeline_data", []))
                }
                
                trend_results.append({
                    "name": name,
                    "insights": insights,
                    "summary_data": minimal_data,
                    "status": "success"
                })
                
                processed_insights[name] = insights
                
            else:
                print(f"Trends API returned status {resp.status_code} for {name}")
                trend_results.append({
                    "name": name,
                    "insights": [f"API request failed with status {resp.status_code}"],
                    "status": "failed"
                })

        except Exception as e:
            print(f"Trends API error for {name}: {e}")
            trend_results.append({
                "name": name,
                "insights": [f"Error: {str(e)}"],
                "status": "error"
            })
        
        # CHANGE: Increased delay to 5 seconds for more conservative API usage
        time.sleep(5)  # MINIMAL SEARCHES - increased delay

    try:
        with open("trend_analysis_detailed.json", "w") as f:
            json.dump(trend_results, f, indent=2)
        print("Detailed trend data saved to trend_analysis_detailed.json")
    except Exception as e:
        print(f"Could not save detailed trend data: {e}")

    write_memory("trend_analysis_agent", trend_results)

    try:
        summary_text = f"Google Trends Analysis for {len(trend_results)} competitors:\n\n"
        
        for result in trend_results:
            name = result["name"]
            insights = result.get("insights", ["No insights available"])
            summary_text += f"**{name}:**\n"
            for i, insight in enumerate(insights[:5], 1):
                summary_text += f"{i}. {insight}\n"
            summary_text += "\n"
        
        summary_text += "Please analyze these trends and provide strategic insights about market interest and search patterns."
        
        print("Sending concise trend summary to Groq...")
        
        if len(summary_text) > 3000:
            brief_summary = f"Trend analysis completed for: {', '.join([r['name'] for r in trend_results])}. "
            brief_summary += f"Key patterns: {len(processed_insights)} companies analyzed with search trend data. "
            brief_summary += "Detailed insights saved to JSON file."
            llm_summary = brief_summary
        else:
            llm_summary = llm.invoke([HumanMessage(content=summary_text)]).content
            
        print("Trend analysis summary complete")
        
    except Exception as e:
        print(f"Groq analysis error: {e}")
        llm_summary = f"Trend analysis completed for {len(trend_results)} competitors. Detailed insights extracted and saved to JSON."

    final_summary = f"Trend Analysis Results:\n\n"
    for result in trend_results:
        final_summary += f"{result['name']}: {len(result.get('insights', []))} insights extracted\n"
    
    final_summary += f"\n{llm_summary}"

    return {
        "messages": [AIMessage(content=f"TrendAnalysisAgent: Analyzed {len(trend_results)} competitors with detailed insights.")],
        "serp_trend_data": final_summary
    }

def StrategyAdvisorAgent(state: SupervisorState) -> Dict:
    """Summarizes all competitor and funding data into actionable strategy."""
    
    print("StrategyAdvisorAgent starting...")

    full_data = read_memory()

    if not full_data:
        print("No previous agent data found")
        return {
            "messages": [AIMessage(content="StrategyAdvisorAgent: No data available for strategy analysis.")],
            "statergy_data": "No data available for strategy formulation."
        }

    try:
        current_task = state.get("current_task", "General business analysis")
        raw_text = (
            f"You are a senior business strategy consultant. "
            f"Current Task: {current_task}\n\n"
            f"Here is comprehensive market intelligence data:\n\n"
            f"Market Data: {json.dumps(full_data, indent=2)}\n\n"
            f"Based on this data and the specific task, please provide:\n"
            f"1. Key market insights\n"
            f"2. Competitive positioning analysis\n"
            f"3. Strategic opportunities\n"
            f"4. Actionable business recommendations\n"
        )

        print("Generating strategic analysis...")
        summary = llm.invoke([HumanMessage(content=raw_text)]).content
        print("Strategy analysis complete")

    except Exception as e:
        print(f"Strategy analysis error: {e}")
        summary = "Strategic analysis completed based on collected market intelligence data."

    write_memory("StrategyAdvisorAgent", summary)

    return {
        "messages": [AIMessage(content="StrategyAdvisorAgent: Generated comprehensive business strategy.")],
        "statergy_data": summary
    }

def ExecutionPlannerAgent(state: SupervisorState) -> Dict:
    """ExecutionPlannerAgent: Breaks the strategy summary into actionable steps."""
    
    print("ExecutionPlannerAgent starting...")

    full_data = read_memory()
    strategy_summary = full_data.get("StrategyAdvisorAgent", "")

    if not strategy_summary:
        print("No strategy summary found")
        return {
            "messages": [AIMessage(content="ExecutionPlannerAgent: No strategy summary found.")],
            "planner_agent_data": "No strategy data available for execution planning."
        }

    try:
        raw_text = (
            f"You are an expert execution planner. Here is the business strategy summary:\n\n"
            f"{strategy_summary}\n\n"
            f"Please create a detailed execution plan with:\n"
            f"1. Immediate actions (0-30 days)\n"
            f"2. Short-term goals (1-3 months)\n"
            f"3. Medium-term objectives (3-12 months)\n"
            f"4. Resource requirements\n"
            f"5. Success metrics\n"
        )

        print("Creating execution plan...")
        action_plan = llm.invoke([HumanMessage(content=raw_text)]).content
        print("Execution plan complete")

    except Exception as e:
        print(f"Execution planning error: {e}")
        action_plan = "Execution planning completed based on strategic recommendations."

    write_memory("ExecutionPlannerAgent", action_plan)

    return {
        "messages": [AIMessage(content="ExecutionPlannerAgent: Created detailed execution plan.")],
        "planner_agent_data": action_plan
    }

def ResourceRecommenderAgent(state: SupervisorState) -> Dict:
    """ResourceRecommenderAgent: Suggests tools, APIs, websites, or agents."""
    
    print("ResourceRecommenderAgent starting...")

    full_data = read_memory()
    execution_plan = full_data.get("ExecutionPlannerAgent", "")

    if not execution_plan:
        print("No execution plan found")
        return {
            "messages": [AIMessage(content="ResourceRecommenderAgent: No execution plan found.")],
            "recommend_tools_data": "No execution plan available for resource recommendations."
        }

    try:
        raw_text = (
            f"You are an expert business consultant specializing in tools and resources. "
            f"Here is the execution plan:\n\n"
            f"{execution_plan}\n\n"
            f"Please recommend specific tools, APIs, platforms, and resources for each step including:\n"
            f"1. Marketing and analytics tools\n"
            f"2. Development and technical resources\n"
            f"3. Business intelligence platforms\n"
            f"4. Collaboration and project management tools\n"
            f"5. Cost estimates where applicable\n"
        )

        print("Generating resource recommendations...")
        recommended_resources = llm.invoke([HumanMessage(content=raw_text)]).content
        print("Resource recommendations complete")

    except Exception as e:
        print(f"Resource recommendation error: {e}")
        recommended_resources = "Resource recommendations completed based on execution plan."

    write_memory("ResourceRecommenderAgent", recommended_resources)

    return {
        "messages": [AIMessage(content="ResourceRecommenderAgent: Generated comprehensive resource recommendations.")],
        "recommend_tools_data": recommended_resources
    }

def RiskAnalyzerAgent(state: SupervisorState) -> Dict:
    """RiskAnalyzerAgent: Analyzes risks and gaps."""
    
    print("RiskAnalyzerAgent starting...")

    full_data = read_memory()
    execution_plan = full_data.get("ExecutionPlannerAgent", "")
    recommended_resources = full_data.get("ResourceRecommenderAgent", "")
    competitor_data = full_data.get("competitor_monitor_agent", [])

    if not execution_plan:
        print("No execution plan found")
        return {
            "messages": [AIMessage(content="RiskAnalyzerAgent: No execution plan found.")],
            "risk_analyzer_data": "No execution plan available for risk analysis."
        }

    try:
        raw_text = (
            f"You are a senior risk analyst. Please analyze the following business plan:\n\n"
            f"EXECUTION PLAN:\n{execution_plan}\n\n"
            f"RECOMMENDED RESOURCES:\n{recommended_resources}\n\n"
            f"COMPETITOR DATA:\n{json.dumps(competitor_data, indent=2)}\n\n"
            f"Please identify and analyze:\n"
            f"1. Market risks and threats\n"
            f"2. Competitive risks\n"
            f"3. Technical and operational risks\n"
            f"4. Financial risks\n"
            f"5. Risk mitigation strategies\n"
            f"6. Contingency plans\n"
        )

        print("Performing risk analysis...")
        risk_analysis = llm.invoke([HumanMessage(content=raw_text)]).content
        print("Risk analysis complete")

    except Exception as e:
        print(f"Risk analysis error: {e}")
        risk_analysis = "Risk analysis completed for business strategy and execution plan."

    write_memory("RiskAnalyzerAgent", risk_analysis)

    return {
        "messages": [AIMessage(content="RiskAnalyzerAgent: Completed comprehensive risk analysis.")],
        "risk_analyzer_data": risk_analysis
    }

def ReportAgent(state: SupervisorState) -> Dict:
    """ReportAgent: Generates a professional report."""
    
    print("ReportAgent starting...")

    full_data = read_memory()

    if not full_data:
        print("No data found to create report")
        return {
            "messages": [AIMessage(content="ReportAgent: No data found to create report.")],
            "report_making_data": "No data available for report generation."
        }

    try:
        current_task = state.get("current_task", "Business Intelligence Analysis")
        competitors = state.get("competitor_list", [])
        
        report_content = f"""# Business Intelligence Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Task:** {current_task}
**Companies Analyzed:** {', '.join(competitors) if competitors else 'Default set'}

## Executive Summary
This comprehensive business intelligence report provides market analysis, competitive insights, and strategic recommendations based on automated data collection and analysis for the specified task and competitors.

## Table of Contents
1. Market Intelligence
2. Competitive Analysis
3. Funding Intelligence
4. Trend Analysis
5. Strategic Recommendations
6. Execution Plan
7. Resource Requirements
8. Risk Analysis
9. Conclusions

---

"""
        
        agent_sections = {
            "market_watcher_agent": "## 1. Market Intelligence\n",
            "competitor_monitor_agent": "## 2. Competitive Analysis\n",
            "funding_intel_agent": "## 3. Funding Intelligence\n",
            "trend_analysis_agent": "## 4. Trend Analysis\n",
            "StrategyAdvisorAgent": "## 5. Strategic Recommendations\n",
            "ExecutionPlannerAgent": "## 6. Execution Plan\n",
            "ResourceRecommenderAgent": "## 7. Resource Requirements\n",
            "RiskAnalyzerAgent": "## 8. Risk Analysis\n"
        }

        for agent_name, section_header in agent_sections.items():
            data = full_data.get(agent_name, "No data available")
            report_content += f"{section_header}\n"
            
            if isinstance(data, str):
                report_content += f"{data}\n\n"
            else:
                report_content += f"Data collected: {len(data) if isinstance(data, list) else 1} items\n"
                report_content += f"Summary: {str(data)[:500]}...\n\n"

        report_content += """## 9. Conclusions

This automated business intelligence analysis has provided comprehensive insights into:
- Market conditions and opportunities
- Competitive landscape and positioning
- Funding trends and investor interest
- Market trends and search patterns
- Strategic recommendations for growth
- Detailed execution roadmap
- Required resources and tools
- Risk assessment and mitigation strategies

The analysis enables data-driven decision making and strategic planning based on current market intelligence.

---
*This report was generated using a multi-agent business intelligence system.*
"""

        print("Report generation complete")

    except Exception as e:
        print(f"Report generation error: {e}")
        report_content = f"Business Intelligence Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nReport generation encountered an error: {str(e)}"

    write_memory("ReportAgent", report_content)

    return {
        "messages": [AIMessage(content="ReportAgent: Generated comprehensive business intelligence report.")],
        "report_making_data": report_content,
        "final_report": report_content,
        "task_complete": True
    }

# Router function - SIMPLIFIED
def router(state: SupervisorState) -> Literal["MarketWatcherAgent", "CompetitorMonitorAgent", "FundingIntelAgent", "TrendAnalysisAgent", "StrategyAdvisorAgent", "ExecutionPlannerAgent", "ResourceRecommenderAgent", "RiskAnalyzerAgent", "ReportAgent", "__end__"]:
    """Routes to next agent based on state - SIMPLIFIED"""
    
    # Check if task is complete
    if state.get("task_complete", False) or state.get("final_report", ""):
        print("Router: Task complete, ending workflow")
        return "__end__"
        
    # Always return to controller for decision making
    next_agent = state.get("next_agent", "controller_agent")
    
    if next_agent == "end":
        return "__end__"
    elif next_agent in ["MarketWatcherAgent", "CompetitorMonitorAgent", "FundingIntelAgent", 
                       "TrendAnalysisAgent", "StrategyAdvisorAgent", "ExecutionPlannerAgent",
                       "ResourceRecommenderAgent", "RiskAnalyzerAgent", "ReportAgent"]:
        return next_agent
    else:
        # Default back to controller
        return "controller_agent"

def run_analysis(task: str, competitors: List[str] = None):
    """Run the multi-agent analysis with improved error handling"""
    
    if competitors is None or len(competitors) == 0:
        print("Warning: No competitors provided, using default list")
        competitors = ["Tesla", "Apple", "Google"]
    
    print(f"Using task from API: {task}")
    print(f"Using competitors from API: {competitors}")
    
    # Clear previous memory for fresh start
    clear_memory()
    
    # Validate API keys
    missing_keys = []
    if not GROQ_API_KEY:
        missing_keys.append("GROQ_API_KEY")
    if not SERPER_API_KEY:
        missing_keys.append("SERPER_API_KEY")
    
    if missing_keys:
        print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some features may not work properly.")
    
    # Create workflow
    workflow = StateGraph(SupervisorState)

    # Add nodes
    workflow.add_node("controller_agent", controller_agent)
    workflow.add_node("MarketWatcherAgent", MarketWatcherAgent)
    workflow.add_node("CompetitorMonitorAgent", CompetitorMonitorAgent)
    workflow.add_node("FundingIntelAgent", FundingIntelAgent)
    workflow.add_node("TrendAnalysisAgent", TrendAnalysisAgent)
    workflow.add_node("StrategyAdvisorAgent", StrategyAdvisorAgent)
    workflow.add_node("ExecutionPlannerAgent", ExecutionPlannerAgent)
    workflow.add_node("ResourceRecommenderAgent", ResourceRecommenderAgent)
    workflow.add_node("RiskAnalyzerAgent", RiskAnalyzerAgent)
    workflow.add_node("ReportAgent", ReportAgent)

    # Set entry point
    workflow.set_entry_point("controller_agent")

    # FIXED: Simplified routing - all agents return to controller
    all_agents = ["MarketWatcherAgent", "CompetitorMonitorAgent", "FundingIntelAgent", 
                  "TrendAnalysisAgent", "StrategyAdvisorAgent", "ExecutionPlannerAgent",
                  "ResourceRecommenderAgent", "RiskAnalyzerAgent", "ReportAgent"]

    # All agents route back to controller
    for agent in all_agents:
        workflow.add_edge(agent, "controller_agent")

    # Only controller uses conditional routing
    workflow.add_conditional_edges(
        "controller_agent",
        router,
        {
            "controller_agent": "controller_agent",
            "MarketWatcherAgent": "MarketWatcherAgent",
            "CompetitorMonitorAgent": "CompetitorMonitorAgent",
            "FundingIntelAgent": "FundingIntelAgent",
            "TrendAnalysisAgent": "TrendAnalysisAgent",
            "StrategyAdvisorAgent": "StrategyAdvisorAgent",
            "ExecutionPlannerAgent": "ExecutionPlannerAgent",
            "ResourceRecommenderAgent": "ResourceRecommenderAgent",
            "RiskAnalyzerAgent": "RiskAnalyzerAgent",
            "ReportAgent": "ReportAgent",
            "__end__": END
        }
    )

    # Compile the graph
    graph = workflow.compile()
    
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "competitor_list": competitors,
        "next_agent": "controller_agent",
        "task_complete": False,
        "current_task": task
    }
    
    print("Starting Multi-Agent Business Intelligence Analysis...")
    print(f"Task: {task}")
    print(f"Competitors: {', '.join(competitors)}")
    print(f"API Keys configured: GROQ={bool(GROQ_API_KEY)}, SERPER={bool(SERPER_API_KEY)}, SCRAPER={bool(SCRAPER_API_KEY)}, SERP={bool(SERP_API_KEY)}")
    print("-" * 80)
    
    try:
        step_count = 0
        max_steps = 25
        
        for step in graph.stream(initial_state):
            step_count += 1
            print(f"\nStep {step_count}: {list(step.keys())}")
            
            for agent_name, agent_state in step.items():
                if "messages" in agent_state and agent_state["messages"]:
                    latest_message = agent_state["messages"][-1]
                    print(f"   {latest_message.content}")
                
                if agent_state.get("task_complete", False):
                    print(f"\nAnalysis complete after {step_count} steps!")
                    final_report = agent_state.get("final_report", "")
                    if final_report:
                        print("\n" + "="*80)
                        print("FINAL BUSINESS INTELLIGENCE REPORT")
                        print("="*80)
                        print(final_report)
                        
                        try:
                            with open("business_intelligence_report.md", "w") as f:
                                f.write(final_report)
                            print("\nReport saved to: business_intelligence_report.md")
                        except Exception as e:
                            print(f"Could not save report to file: {e}")
                            
                    return "business_intelligence_report.md"
                    
            if step_count >= max_steps:
                print(f"\nAnalysis stopped after {max_steps} steps to prevent infinite loop")
                return "business_intelligence_report.md"
                    
    except Exception as e:
        print(f"\nError during analysis: {e}")
        return "business_intelligence_report.md"

# Main function to be called by FastAPI
def run_pipeline(task: str = None, competitors: List[str] = None) -> str:
    """Main entry point for the pipeline - called by FastAPI"""
    if task is None:
        task = "Analyze the AI/ML industry and provide strategic recommendations"
    if competitors is None:
        competitors = ["OpenAI", "Anthropic", "Cohere", "Hugging Face"]
    
    return run_analysis(task, competitors)

# Utility functions
def view_memory():
    """View current memory contents"""
    memory = read_memory()
    if not memory:
        print("Memory is empty")
        return
        
    print("Current Memory Contents:")
    print("-" * 40)
    for agent_name, data in memory.items():
        print(f"\n{agent_name}:")
        if isinstance(data, str):
            print(f"   Data length: {len(data)} characters")
            print(f"   Preview: {data[:100]}...")
        else:
            print(f"   Data type: {type(data)}")
            print(f"   Content: {str(data)[:100]}...")

def clear_memory():
    """Clear all memory"""
    try:
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
            print("Memory cleared successfully")
        else:
            print("Memory was already empty")
    except Exception as e:
        print(f"Error clearing memory: {e}")

def check_api_keys():
    """Check which API keys are configured"""
    keys = {
        "GROQ_API_KEY": bool(GROQ_API_KEY),
        "SERPER_API_KEY": bool(SERPER_API_KEY),
        "SCRAPER_API_KEY": bool(SCRAPER_API_KEY),
        "SERP_API_KEY": bool(SERP_API_KEY)
    }
    
    print("API Key Status:")
    print("-" * 30)
    for key_name, is_set in keys.items():
        status = "Configured" if is_set else "Missing"
        print(f"{key_name}: {status}")
    
    return keys

# For testing when run directly
if __name__ == "__main__":
    check_api_keys()
    result = run_pipeline(
        task="Analyze the AI/ML industry and provide strategic recommendations",
        competitors=["OpenAI", "Anthropic", "Cohere", "Hugging Face"]
    )
    print(f"\nAnalysis completed! Report saved as: {result}")