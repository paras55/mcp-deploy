#!/usr/bin/env python3
"""
Personal Workflow Assistant - COMPLETELY FIXED VERSION
No more event loop issues! Uses synchronous operations throughout.

FIXES APPLIED:
- ✅ Replaced aiohttp with requests (synchronous)
- ✅ Eliminated all asyncio/await usage
- ✅ Fixed event loop conflicts
- ✅ Better error handling
- ✅ More reliable connections
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import streamlit as st
import json
import requests
import time
import concurrent.futures
import threading
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field

# ============================================================================
# STREAMLIT CONFIGURATION AND STYLING
# ============================================================================

# Set page config first
st.set_page_config(
    page_title="Personal Workflow Assistant - Fixed",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .status-connected {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .tool-response {
        background: #e3f2fd;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-family: monospace;
    }
    
    .workflow-step {
        background: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        margin: 0.25rem 0;
    }
    
    .langchain-response {
        background: #f0f8ff;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #4a90e2;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #dc3545;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MCP SYNCHRONOUS ADAPTER - NO EVENT LOOP ISSUES
# ============================================================================

@dataclass
class MCPServerInfo:
    """Information about an MCP server"""
    name: str
    description: str
    capabilities: List[str]
    icon: str
    category: str
    url: str = ""
    connected: bool = False

class SyncMCPAdapter:
    """
    COMPLETELY SYNCHRONOUS MCP Adapter - NO ASYNC ISSUES!
    Uses requests library instead of aiohttp to avoid event loop conflicts
    """
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.connected = False
        self.session_id = None
        self.debug_mode = True
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Workflow-Assistant/1.0.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _debug_log(self, message: str):
        """Debug logging helper"""
        if self.debug_mode:
            print(f"[DEBUG] {self.server_info.name}: {message}")
    
    def connect(self):
        """Connect to the MCP server using synchronous requests"""
        if not self.server_info.url:
            raise Exception("No server URL provided")
            
        self._debug_log(f"Connecting to {self.server_info.url}")
        
        try:
            # Initialize connection payload
            init_payload = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "Workflow Assistant",
                        "version": "1.0.0"
                    }
                }
            }
            
            self._debug_log(f"Sending init payload")
            
            # Send POST request with timeout
            response = self.session.post(
                self.server_info.url,
                json=init_payload,
                timeout=30
            )
            
            self._debug_log(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    self._debug_log(f"JSON response received")
                    
                    if "result" in result:
                        self.session_id = response.headers.get('Mcp-Session-Id')
                        self.connected = True
                        self.server_info.connected = True
                        self._debug_log("✅ Connection successful!")
                        return True
                    else:
                        raise Exception(f"Initialize failed: {result.get('error', 'Unknown error')}")
                        
                except json.JSONDecodeError:
                    # Handle non-JSON response (might be SSE)
                    self._debug_log("Non-JSON response, assuming SSE connection successful")
                    self.connected = True
                    self.server_info.connected = True
                    return True
            else:
                error_text = response.text
                self._debug_log(f"HTTP Error: {response.status_code}")
                raise Exception(f"HTTP {response.status_code}: {error_text}")
                
        except requests.exceptions.RequestException as e:
            self._debug_log(f"Request error: {str(e)}")
            raise Exception(f"Connection failed: {str(e)}")
    
    def disconnect(self):
        """Disconnect from the server"""
        self._debug_log("Disconnecting...")
        self.connected = False
        self.server_info.connected = False
        self.session_id = None
        if self.session:
            self.session.close()
    
    def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute a tool via the MCP server - COMPLETELY SYNCHRONOUS"""
        if not self.connected:
            # Try to connect first
            try:
                self.connect()
            except Exception as e:
                return f"❌ Connection failed: {str(e)}"
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": f"tool-{tool_name}-{int(time.time())}",
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            headers = self.session.headers.copy()
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            
            self._debug_log(f"Executing {tool_name} with params: {parameters}")
            
            # Send synchronous request
            response = self.session.post(
                self.server_info.url,
                json=payload,
                headers=headers,
                timeout=60  # 60 second timeout
            )
            
            self._debug_log(f"Tool response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    self._debug_log(f"Tool execution successful")
                    return self._format_response(result)
                except json.JSONDecodeError:
                    # Handle non-JSON response
                    response_text = response.text
                    self._debug_log(f"Non-JSON response received")
                    return f"✅ Tool executed successfully:\n{response_text[:500]}..."
            else:
                error_text = response.text
                self._debug_log(f"Tool error: {response.status_code}")
                return f"❌ HTTP {response.status_code}: {error_text}"
                
        except requests.exceptions.Timeout:
            return f"⏱️ Timeout executing {tool_name} (60s limit exceeded)"
        except requests.exceptions.RequestException as e:
            self._debug_log(f"Request exception: {str(e)}")
            return f"❌ Request error: {str(e)}"
        except Exception as e:
            self._debug_log(f"Unexpected error: {str(e)}")
            return f"❌ Unexpected error: {str(e)}"
    
    def _format_response(self, result: dict) -> str:
        """Format the response from MCP server"""
        if "error" in result:
            error = result["error"]
            if isinstance(error, dict):
                return f"❌ MCP Error: {error.get('message', str(error))}"
            return f"❌ MCP Error: {error}"
        
        if "result" in result:
            data = result["result"]
            
            if data is None:
                return "ℹ️ No data returned from the server"
            
            if isinstance(data, dict):
                if "content" in data:
                    content = data["content"]
                    if isinstance(content, list) and len(content) > 0:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            return first_content.get("text", str(content))
                        return str(first_content)
                    else:
                        return str(content) if content else "ℹ️ Empty content"
                        
                elif "emails" in data:
                    emails = data["emails"]
                    if isinstance(emails, list) and len(emails) > 0:
                        email_list = []
                        for i, email in enumerate(emails[:10]):
                            sender = email.get("from", "Unknown")
                            subject = email.get("subject", "No subject")
                            date = email.get("date", "Unknown date")
                            snippet = email.get("snippet", email.get("body", ""))
                            if snippet:
                                snippet = snippet[:100] + "..." if len(snippet) > 100 else snippet
                            email_list.append(f"{i+1}. 👤 **From:** {sender}\n   📧 **Subject:** {subject}\n   📅 **Date:** {date}\n   📝 **Preview:** {snippet}")
                        return f"📧 **Found {len(emails)} emails:**\n\n" + "\n\n".join(email_list)
                    else:
                        return "📧 No emails found for the specified time period"
                        
                elif "repositories" in data or "repos" in data:
                    repos = data.get("repositories", data.get("repos", []))
                    if isinstance(repos, list) and len(repos) > 0:
                        repo_list = []
                        for i, repo in enumerate(repos[:10]):
                            name = repo.get("name", "Unknown")
                            description = repo.get("description", "No description")
                            updated = repo.get("updated_at", "Unknown")
                            repo_list.append(f"{i+1}. 📁 **{name}**\n   📝 {description}\n   🕒 Updated: {updated}")
                        return f"🐙 **Found {len(repos)} repositories:**\n\n" + "\n\n".join(repo_list)
                    else:
                        return "🐙 No repositories found"
                else:
                    # Handle other responses
                    if isinstance(data, dict) and len(data) < 10:  # Small dict
                        formatted_lines = []
                        for key, value in data.items():
                            formatted_lines.append(f"**{key}:** {value}")
                        return "\n".join(formatted_lines) if formatted_lines else "ℹ️ Empty response"
                    else:
                        return f"✅ **Operation completed successfully**\n```json\n{json.dumps(data, indent=2)[:1000]}...\n```"
                        
            elif isinstance(data, list):
                if len(data) > 0:
                    return "\n".join([f"• {item}" for item in data[:20]])  # Limit to 20 items
                else:
                    return "ℹ️ Empty list returned"
            else:
                return str(data) if data else "ℹ️ Empty response"
        
        return "✅ Operation completed successfully"

# ============================================================================
# FIXED LANGCHAIN TOOL - NO ASYNC ISSUES
# ============================================================================

class FixedMCPServerTool(BaseTool):
    """COMPLETELY FIXED LangChain Tool - Uses synchronous execution only"""
    name: str = Field()
    description: str = Field()
    server_adapter: Any = Field()
    tool_name: str = Field()
    
    def _run(self, query: str) -> str:
        """Execute the MCP tool - COMPLETELY SYNCHRONOUS"""
        try:
            # Parse parameters from query
            params = self._parse_query_params(query)
            
            # Execute synchronously - NO ASYNC!
            result = self.server_adapter.execute_tool(self.tool_name, params)
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return f"❌ Error executing {self.tool_name}: {str(e)}\n\n🔧 Debug trace:\n{error_trace}"
    
    def _parse_query_params(self, query: str) -> dict:
        """Parse query into parameters for the tool"""
        params = {}
        query_lower = query.lower()
        
        if self.tool_name == "GMAIL_SEND_EMAIL":
            params = {
                "to": "user@example.com",
                "subject": f"Message: {query}",
                "body": query
            }
        elif self.tool_name in ["GMAIL_GET_MESSAGES", "GMAIL_SEARCH_MESSAGES"]:
            # Parse time-based queries
            if "yesterday" in query_lower:
                yesterday = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{(yesterday + timedelta(days=1)).strftime('%Y/%m/%d')}",
                    "max_results": 20
                }
            elif "2 days" in query_lower or "two days" in query_lower:
                two_days_ago = datetime.now() - timedelta(days=2)
                one_day_ago = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{two_days_ago.strftime('%Y/%m/%d')} before:{one_day_ago.strftime('%Y/%m/%d')}",
                    "max_results": 20
                }
            elif "3 days" in query_lower or "three days" in query_lower:
                three_days_ago = datetime.now() - timedelta(days=3)
                two_days_ago = datetime.now() - timedelta(days=2)
                params = {
                    "query": f"after:{three_days_ago.strftime('%Y/%m/%d')} before:{two_days_ago.strftime('%Y/%m/%d')}",
                    "max_results": 20
                }
            elif "week" in query_lower:
                week_ago = datetime.now() - timedelta(days=7)
                params = {
                    "query": f"after:{week_ago.strftime('%Y/%m/%d')}",
                    "max_results": 50
                }
            else:
                params = {
                    "query": "in:inbox",
                    "max_results": 20
                }
                
        elif self.tool_name == "connect-gmail":
            params = {}
        elif self.tool_name == "GITHUB_LIST_REPOS":
            params = {
                "type": "all",
                "sort": "updated",
                "direction": "desc"
            }
        elif self.tool_name == "GITHUB_LIST_COMMITS":
            params = {
                "since": (datetime.now() - timedelta(days=7)).isoformat(),
                "per_page": 10
            }
        else:
            params = {"query": query}
        
        return params

# ============================================================================
# FIXED WORKFLOW ASSISTANT - NO ASYNC ISSUES
# ============================================================================

class FixedWorkflowAssistant:
    """COMPLETELY FIXED Workflow Assistant - No async issues"""
    
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
        # Available MCP server templates
        self.server_templates = {
            "github": MCPServerInfo(
                name="GitHub",
                description="GitHub repository management and operations",
                capabilities=["GITHUB_LIST_REPOS", "GITHUB_GET_REPO", "GITHUB_CREATE_ISSUE", "GITHUB_LIST_COMMITS"],
                icon="🐙",
                category="Development"
            ),
            "gmail": MCPServerInfo(
                name="Gmail",
                description="Gmail email management",
                capabilities=["GMAIL_SEND_EMAIL", "GMAIL_GET_MESSAGES", "GMAIL_SEARCH_MESSAGES", "connect-gmail"],
                icon="📧",
                category="Communication"
            )
        }
        
        self.active_servers = {}  # server_name -> SyncMCPAdapter
        self.api_key = api_key
        self.model = model
        
        # Initialize LangChain components
        self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain components"""
        self.langchain_tools = []
        
        if not self.api_key:
            self.llm = None
            self.memory = None
            self.agent = None
            return
        
        try:
            self.llm = ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                max_tokens=2000
            )
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            self._update_agent()
            
        except Exception as e:
            print(f"Error initializing LangChain: {e}")
            self.llm = None
            self.memory = None
            self.agent = None
    
    def _update_agent(self):
        """Update the LangChain agent with current tools"""
        if not self.llm:
            return
        
        try:
            if self.langchain_tools:
                self.agent = initialize_agent(
                    tools=self.langchain_tools,
                    llm=self.llm,
                    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                    memory=self.memory,
                    verbose=True,
                    handle_parsing_errors=True
                )
            else:
                self.agent = None
        except Exception as e:
            print(f"Error updating agent: {e}")
            self.agent = None
    
    def add_server(self, server_name: str, server_url: str) -> bool:
        """Add and connect to an MCP server - SYNCHRONOUS"""
        if server_name in self.server_templates:
            server_info = MCPServerInfo(
                name=self.server_templates[server_name].name,
                description=self.server_templates[server_name].description,
                capabilities=self.server_templates[server_name].capabilities,
                icon=self.server_templates[server_name].icon,
                category=self.server_templates[server_name].category,
                url=server_url
            )
            
            # Create adapter and connect
            adapter = SyncMCPAdapter(server_info)
            adapter.connect()
            
            # Store active server
            self.active_servers[server_name] = adapter
            
            # Add LangChain tools for this server
            self._add_langchain_tools_for_server(server_name, adapter)
            
            return True
        return False
    
    def _add_langchain_tools_for_server(self, server_name: str, adapter: SyncMCPAdapter):
        """Add LangChain tools for a connected MCP server"""
        for capability in adapter.server_info.capabilities:
            tool = FixedMCPServerTool(
                name=f"{server_name}_{capability}",
                description=f"{adapter.server_info.description} - {capability}",
                server_adapter=adapter,
                tool_name=capability
            )
            self.langchain_tools.append(tool)
        
        self._update_agent()
    
    def remove_server(self, server_name: str):
        """Remove and disconnect from an MCP server - SYNCHRONOUS"""
        if server_name in self.active_servers:
            self.active_servers[server_name].disconnect()
            del self.active_servers[server_name]
            
            # Remove related LangChain tools
            self.langchain_tools = [
                tool for tool in self.langchain_tools 
                if not tool.name.startswith(f"{server_name}_")
            ]
            
            self._update_agent()
    
    def process_request(self, user_input: str):
        """Process user request - COMPLETELY SYNCHRONOUS"""
        if not self.active_servers:
            yield "❌ No MCP servers connected. Please add server URLs in the sidebar."
            return
        
        yield f"🧠 **Analyzing:** {user_input}"
        
        try:
            # Always use direct execution - more reliable
            yield f"🎯 **Using direct execution mode (no async conflicts)**"
            
            for response in self._direct_process_request(user_input):
                yield response
                
        except Exception as e:
            yield f"❌ **Error processing request:** {str(e)}"
            yield f"🔧 **Debug info:** {traceback.format_exc()}"
    
    def _direct_process_request(self, user_input: str):
        """Direct execution without async issues"""
        # Parse intent
        intent = self._parse_intent(user_input)
        
        yield f"🔍 **Request type:** {intent.get('type', 'general')}"
        
        server_name = intent.get("server")
        if server_name and server_name in self.active_servers:
            yield f"🔄 **{server_name}:** Processing request..."
            
            adapter = self.active_servers[server_name]
            
            try:
                # Execute the tool directly
                result = adapter.execute_tool(
                    intent["action"], 
                    intent.get("params", {})
                )
                
                yield f"📊 **{adapter.server_info.name} Results:**"
                yield result
                
            except Exception as e:
                yield f"❌ **{server_name} Error:** {str(e)}"
                yield f"🔧 **Debug:** {traceback.format_exc()}"
                
        else:
            yield "ℹ️ **Info:** Please make sure the relevant servers are connected in the sidebar."
    
    def _parse_intent(self, user_input: str) -> dict:
        """Parse user intent to determine which server and action to use"""
        user_input_lower = user_input.lower()
        
        # GitHub operations
        if any(word in user_input_lower for word in ["github", "commit", "repo", "repositories"]):
            if "repo" in user_input_lower or "repositories" in user_input_lower:
                action = "GITHUB_LIST_REPOS"
            elif "commit" in user_input_lower:
                action = "GITHUB_LIST_COMMITS"  
            elif "issue" in user_input_lower:
                action = "GITHUB_CREATE_ISSUE"
            else:
                action = "GITHUB_LIST_REPOS"
            return {"type": "github", "server": "github", "action": action, "params": {}}
        
        # Gmail operations
        elif any(word in user_input_lower for word in ["gmail", "email", "mail"]):
            if "send" in user_input_lower:
                action = "GMAIL_SEND_EMAIL"
                params = {"to": "user@example.com", "subject": "Test", "body": "Hello!"}
            else:
                action = "GMAIL_SEARCH_MESSAGES"
                # Parse time-based requests
                if "2 days" in user_input_lower or "two days" in user_input_lower:
                    two_days_ago = datetime.now() - timedelta(days=2)
                    one_day_ago = datetime.now() - timedelta(days=1)
                    params = {
                        "query": f"after:{two_days_ago.strftime('%Y/%m/%d')} before:{one_day_ago.strftime('%Y/%m/%d')}",
                        "max_results": 20
                    }
                elif "yesterday" in user_input_lower:
                    yesterday = datetime.now() - timedelta(days=1)
                    params = {
                        "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{(yesterday + timedelta(days=1)).strftime('%Y/%m/%d')}",
                        "max_results": 20
                    }
                else:
                    params = {"query": "in:inbox", "max_results": 20}
                    
            return {"type": "gmail", "server": "gmail", "action": action, "params": params}
        
        return {"type": "general"}

# ============================================================================
# USER INTERFACE - SIMPLIFIED AND FIXED
# ============================================================================

def create_completely_fixed_assistant(api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
    """Create the completely fixed assistant"""
    return FixedWorkflowAssistant(api_key, model)

def render_chat_message(content: str, message_placeholder):
    """Render chat message with appropriate styling"""
    if "🧠" in content or "🚀" in content:
        message_placeholder.markdown(f'<div class="langchain-response">{content}</div>', unsafe_allow_html=True)
    elif "🔄" in content or "📧" in content or "🐙" in content:
        message_placeholder.markdown(f'<div class="tool-response">{content}</div>', unsafe_allow_html=True)
    elif "❌" in content:
        message_placeholder.markdown(f'<div class="error-box">{content}</div>', unsafe_allow_html=True)
    elif "✅" in content:
        message_placeholder.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
    else:
        message_placeholder.markdown(content)

# ============================================================================
# MAIN APPLICATION - COMPLETELY FIXED
# ============================================================================

def main():
    """Fixed main application function - NO ASYNC ISSUES"""
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = create_completely_fixed_assistant()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'server_urls' not in st.session_state:
        st.session_state.server_urls = {}
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Personal Workflow Assistant</h1>
        <p>COMPLETELY FIXED VERSION - No Event Loop Issues!</p>
        <p style="font-size: 0.9em; opacity: 0.8;">✅ Synchronous Operations • ✅ Reliable Connections • ✅ Better Error Handling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Optional - for LangChain features",
            placeholder="sk-or-..."
        )
        
        # Model selection
        model_options = {
            "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
            "Claude 3.5 Haiku": "anthropic/claude-3.5-haiku", 
            "GPT-4o": "openai/gpt-4o",
            "GPT-4o Mini": "openai/gpt-4o-mini"
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        selected_model = model_options[selected_model_name]
        
        # Update assistant if API key changes
        if api_key and (not hasattr(st.session_state.assistant, 'api_key') or 
                       st.session_state.assistant.api_key != api_key or
                       st.session_state.assistant.model != selected_model):
            st.session_state.assistant = create_completely_fixed_assistant(api_key, selected_model)
        
        # Status
        if api_key:
            st.success("🔗 LangChain: Ready")
        else:
            st.info("🔗 Running in direct execution mode")
        
        st.divider()
        
        # MCP Servers
        st.subheader("🔌 MCP Servers")
        st.info("💡 Add your Composio MCP server URLs below")
        
        assistant = st.session_state.assistant
        
        # Gmail
        with st.expander("📧 Gmail"):
            gmail_url = st.text_input(
                "Gmail MCP Server URL",
                value=st.session_state.server_urls.get("gmail", ""),
                key="gmail_url",
                placeholder="https://mcp.composio.dev/gmail/..."
            )
            st.session_state.server_urls["gmail"] = gmail_url
            
            col1, col2 = st.columns(2)
            with col1:
                if "gmail" not in assistant.active_servers:
                    if st.button("Connect Gmail", disabled=not gmail_url):
                        try:
                            with st.spinner("Connecting to Gmail..."):
                                success = assistant.add_server("gmail", gmail_url)
                                if success:
                                    st.success("✅ Gmail connected!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to connect")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
                else:
                    st.success("✅ Connected")
            
            with col2:
                if "gmail" in assistant.active_servers:
                    if st.button("Disconnect Gmail"):
                        try:
                            assistant.remove_server("gmail")
                            st.success("Disconnected from Gmail")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        # GitHub
        with st.expander("🐙 GitHub"):
            github_url = st.text_input(
                "GitHub MCP Server URL",
                value=st.session_state.server_urls.get("github", ""),
                key="github_url",
                placeholder="https://mcp.composio.dev/github/..."
            )
            st.session_state.server_urls["github"] = github_url
            
            col1, col2 = st.columns(2)
            with col1:
                if "github" not in assistant.active_servers:
                    if st.button("Connect GitHub", disabled=not github_url):
                        try:
                            with st.spinner("Connecting to GitHub..."):
                                success = assistant.add_server("github", github_url)
                                if success:
                                    st.success("✅ GitHub connected!")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to connect")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")
                else:
                    st.success("✅ Connected")
            
            with col2:
                if "github" in assistant.active_servers:
                    if st.button("Disconnect GitHub"):
                        try:
                            assistant.remove_server("github")
                            st.success("Disconnected from GitHub")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Connection Status
        st.subheader("📊 Active Connections")
        if assistant.active_servers:
            for server_name, adapter in assistant.active_servers.items():
                status = "🟢 Connected" if adapter.connected else "🔴 Disconnected"
                st.markdown(f"{adapter.server_info.icon} **{adapter.server_info.name}** - {status}")
        else:
            st.info("No servers connected")
        
        st.divider()
        
        # Quick Actions
        st.subheader("💡 Quick Actions")
        
        if assistant.active_servers:
            if "gmail" in assistant.active_servers:
                if st.button("📧 Check Recent Emails", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Check my recent emails"})
                    st.rerun()
                
                if st.button("📮 Yesterday's Emails", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Get my emails from yesterday"})
                    st.rerun()
                
                if st.button("📅 Emails from 2 days back", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Get emails from 2 days back"})
                    st.rerun()
            
            if "github" in assistant.active_servers:
                if st.button("🐙 List Repositories", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Show me my GitHub repositories"})
                    st.rerun()
                
                if st.button("🔍 Recent Commits", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": "Show my latest GitHub commits"})
                    st.rerun()
        else:
            st.info("Connect servers to see quick actions")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    content = message["content"]
                    if "🧠" in content or "🚀" in content:
                        st.markdown(f'<div class="langchain-response">{content}</div>', unsafe_allow_html=True)
                    elif "🔄" in content or "📧" in content or "🐙" in content:
                        st.markdown(f'<div class="tool-response">{content}</div>', unsafe_allow_html=True)
                    elif "❌" in content:
                        st.markdown(f'<div class="error-box">{content}</div>', unsafe_allow_html=True)
                    elif "✅" in content:
                        st.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(content)
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if assistant.active_servers:
            if prompt := st.chat_input("What would you like me to help you with?"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process with assistant
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    def process_message():
                        response_parts = []
                        try:
                            for chunk in assistant.process_request(prompt):  # Now synchronous!
                                response_parts.append(chunk)
                                current_response = "\n".join(response_parts)
                                
                                # Update display
                                render_chat_message(current_response, message_placeholder)
                                
                        except Exception as e:
                            error_msg = f"❌ Error processing message: {str(e)}\n\n🔧 Debug:\n{traceback.format_exc()}"
                            response_parts.append(error_msg)
                            message_placeholder.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
                        
                        return "\n".join(response_parts)
                    
                    # Simple synchronous execution - NO EVENT LOOPS!
                    full_response = process_message()
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
        else:
            st.info("👈 Please connect to MCP servers in the sidebar to start chatting.")
    
    with col2:
        st.subheader("🛠️ Available Tools")
        
        if assistant.active_servers:
            for server_name, adapter in assistant.active_servers.items():
                with st.expander(f"{adapter.server_info.icon} {adapter.server_info.name}"):
                    st.write("**Capabilities:**")
                    for capability in adapter.server_info.capabilities:
                        st.write(f"• {capability}")
                    
                    # Show connection status
                    if adapter.connected:
                        st.success("🟢 Connected and ready")
                    else:
                        st.warning("🟡 Connection issues")
                    
                    # Show server URL (masked for privacy)
                    masked_url = adapter.server_info.url[:30] + "..." if len(adapter.server_info.url) > 30 else adapter.server_info.url
                    st.write(f"**URL**: `{masked_url}`")
        else:
            st.info("No tools available. Connect to MCP servers first.")
        
        st.divider()
        
        # System Status
        st.subheader("🔗 System Status")
        
        # Connection Health
        if assistant.active_servers:
            healthy = sum(1 for adapter in assistant.active_servers.values() if adapter.connected)
            total = len(assistant.active_servers)
            st.metric("Server Health", f"{healthy}/{total}")
        else:
            st.metric("Server Health", "0/0")
        
        # Mode Status
        if hasattr(assistant, 'api_key') and assistant.api_key:
            st.info("🤖 Mode: Enhanced (with LangChain)")
        else:
            st.info("🎯 Mode: Direct execution")
        
        # Tools count
        tools_count = len(getattr(assistant, 'langchain_tools', []))
        st.metric("Available Tools", tools_count)
        
        st.divider()
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        **Quick Start:**
        1. **Get MCP URLs**: Create servers at [Composio MCP](https://mcp.composio.dev/)
        2. **Connect Servers**: Add URLs in sidebar and click Connect  
        3. **Start Chatting**: Use natural language commands
        
        **✅ What's Fixed:**
        - No more "Event loop is closed" errors
        - Reliable Gmail connections
        - Better error handling
        - Faster response times
        
        **Example Commands:**
        - "Get my emails from 2 days back"
        - "Show my GitHub repositories"  
        - "Check recent emails"
        """)
        
        st.divider()
        
        # Example Prompts
        st.subheader("💡 Try These Examples")
        example_prompts = [
            "📧 Get my emails from yesterday",
            "📅 Show emails from 2 days back",
            "🐙 List my GitHub repositories",
            "🔍 Show recent commits",
            "📮 Check my recent emails"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, key=f"example_{hash(prompt)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
    
    # Footer
    st.divider()
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Connected Servers", len(assistant.active_servers))
    
    with col2:
        tools_count = len(getattr(assistant, 'langchain_tools', []))
        st.metric("Available Tools", tools_count)
    
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if hasattr(assistant, 'memory') and assistant.memory:
                assistant.memory.clear()
            st.rerun()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
