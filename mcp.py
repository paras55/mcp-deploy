#!/usr/bin/env python3
"""
Personal Workflow Assistant - FIXED VERSION
A Streamlit application that integrates MCP (Model Context Protocol) servers with LangChain
for intelligent workflow automation and management.

FIXES:
- ✅ Event loop issues resolved
- ✅ Better error handling
- ✅ Improved async operations
- ✅ More robust MCP connections
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import streamlit as st
import asyncio
import json
import requests
import time
import aiohttp
import logging
import os
import traceback
import concurrent.futures
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field

# ============================================================================
# STREAMLIT CONFIGURATION AND STYLING
# ============================================================================

# Set page config first
st.set_page_config(
    page_title="Personal Workflow Assistant",
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
    
    .feature-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .status-connected {
        background: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .status-disconnected {
        background: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
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
    
    .mcp-server-card {
        background: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #b3d9ff;
        margin: 0.5rem 0;
    }
    
    .server-url-input {
        background: white;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #ddd;
        margin: 0.25rem 0;
        font-family: monospace;
        font-size: 0.85em;
    }
    
    .debug-info {
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #6c757d;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MCP (MODEL CONTEXT PROTOCOL) SECTION
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

class MCPServerAdapter:
    """
    IMPROVED Adapter for communicating with MCP servers using the Streamable HTTP protocol.
    Handles connection, tool listing, and tool execution with better error handling.
    """
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.session = None
        self.connected = False
        self.session_id = None
        self.event_source = None
        self.debug_mode = True  # Enable debug logging
    
    def _debug_log(self, message: str):
        """Debug logging helper"""
        if self.debug_mode:
            print(f"[DEBUG] {self.server_info.name}: {message}")
    
    async def connect(self):
        """Connect to the MCP server using Streamable HTTP protocol"""
        if not self.server_info.url:
            raise Exception("No server URL provided")
            
        self._debug_log(f"Attempting to connect to {self.server_info.url}")
        
        try:
            # Initialize HTTP session with proper MCP headers
            self.session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Workflow-Assistant/1.0.0"
                },
                timeout=aiohttp.ClientTimeout(total=60)  # Increased timeout
            )
            
            # Step 1: Initialize connection with proper MCP protocol
            init_payload = {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "Workflow Assistant",
                        "version": "1.0.0"
                    }
                }
            }
            
            self._debug_log(f"Sending initialize request: {json.dumps(init_payload, indent=2)}")
            
            # Send initialize request with proper headers
            async with self.session.post(
                self.server_info.url,
                json=init_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            ) as response:
                
                self._debug_log(f"Response status: {response.status}")
                self._debug_log(f"Response headers: {dict(response.headers)}")
                
                # Check if response is SSE stream or JSON
                content_type = response.headers.get('Content-Type', '')
                
                if 'text/event-stream' in content_type:
                    # Handle SSE response - read the stream for session ID
                    await self._handle_sse_initialization(response)
                elif 'application/json' in content_type and response.status == 200:
                    # Handle JSON response
                    result = await response.json()
                    self._debug_log(f"JSON response: {json.dumps(result, indent=2)}")
                    if "result" in result:
                        # Extract session ID from headers if present
                        self.session_id = response.headers.get('Mcp-Session-Id')
                        self.connected = True
                        self.server_info.connected = True
                        self._debug_log("Connection successful!")
                        return True
                    else:
                        raise Exception(f"Initialize failed: {result.get('error', 'Unknown error')}")
                else:
                    error_text = await response.text()
                    self._debug_log(f"Error response: {error_text}")
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            self._debug_log(f"Connection error: {str(e)}")
            if self.session:
                await self.session.close()
            raise Exception(f"Connection failed: {str(e)}")
    
    async def _handle_sse_initialization(self, response):
        """Handle SSE stream initialization to extract session info"""
        try:
            self._debug_log("Handling SSE initialization...")
            async for line in response.content:
                line = line.decode('utf-8').strip()
                self._debug_log(f"SSE line: {line}")
                
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    try:
                        event_data = json.loads(data)
                        self._debug_log(f"SSE event: {json.dumps(event_data, indent=2)}")
                        if event_data.get('method') == 'initialize' or 'result' in event_data:
                            # Extract session ID from response headers
                            self.session_id = response.headers.get('Mcp-Session-Id')
                            self.connected = True
                            self.server_info.connected = True
                            self._debug_log("SSE connection successful!")
                            return True
                    except json.JSONDecodeError as e:
                        self._debug_log(f"JSON decode error: {e}")
                        continue
                        
            raise Exception("No valid initialization response received from SSE stream")
            
        except Exception as e:
            self._debug_log(f"SSE initialization error: {str(e)}")
            raise Exception(f"SSE initialization failed: {str(e)}")
    
    async def disconnect(self):
        """Disconnect from the server"""
        self._debug_log("Disconnecting...")
        self.connected = False
        self.server_info.connected = False
        self.session_id = None
        
        if self.event_source:
            self.event_source.close()
            self.event_source = None
            
        if self.session:
            await self.session.close()
            self.session = None
    
    async def list_tools(self) -> List[dict]:
        """List available tools from the MCP server"""
        if not self.connected or not self.session:
            raise Exception("Not connected to server")
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": "tools-list",
                "method": "tools/list",
                "params": {}
            }
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Include session ID if we have one
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            
            self._debug_log(f"Listing tools with payload: {json.dumps(payload, indent=2)}")
            
            async with self.session.post(
                self.server_info.url,
                json=payload,
                headers=headers
            ) as response:
                
                content_type = response.headers.get('Content-Type', '')
                
                if 'text/event-stream' in content_type:
                    # Handle SSE response
                    tools = await self._read_sse_response(response)
                    return tools.get("tools", []) if isinstance(tools, dict) else []
                elif 'application/json' in content_type and response.status == 200:
                    result = await response.json()
                    if "result" in result:
                        return result["result"].get("tools", [])
                    else:
                        return []
                else:
                    return []
        except Exception as e:
            self._debug_log(f"Error listing tools: {e}")
            return []
    
    async def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute a tool via the MCP server with improved error handling"""
        if not self.connected or not self.session:
            raise Exception("Not connected to server")
        
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
            
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            }
            
            # Include session ID if we have one
            if self.session_id:
                headers["Mcp-Session-Id"] = self.session_id
            
            debug_info = f"🔧 Executing {tool_name} with params: {parameters}"
            self._debug_log(debug_info)
            
            async with self.session.post(
                self.server_info.url,
                json=payload,
                headers=headers
            ) as response:
                
                content_type = response.headers.get('Content-Type', '')
                status_info = f"Response status: {response.status}, content-type: {content_type}"
                self._debug_log(status_info)
                
                if 'text/event-stream' in content_type:
                    # Handle SSE response
                    result = await self._read_sse_response(response)
                    formatted_result = self._format_response({"result": result})
                    return formatted_result
                elif 'application/json' in content_type and response.status == 200:
                    result = await response.json()
                    self._debug_log(f"Tool execution result: {json.dumps(result, indent=2)}")
                    formatted_result = self._format_response(result)
                    return formatted_result
                else:
                    error_text = await response.text()
                    self._debug_log(f"Error response: {error_text}")
                    return f"❌ Error Response: {error_text}"
                    
        except asyncio.TimeoutError:
            return f"⏱️ Timeout executing {tool_name} (60s limit exceeded)"
        except Exception as e:
            error_msg = f"❌ Error executing {tool_name}: {str(e)}"
            self._debug_log(error_msg)
            return error_msg
    
    async def _read_sse_response(self, response):
        """Read and parse SSE stream response with better error handling"""
        try:
            result_data = None
            line_count = 0
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                line_count += 1
                
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    
                    # Skip empty data or comments
                    if not data or data.startswith(':'):
                        continue
                        
                    try:
                        event_data = json.loads(data)
                        self._debug_log(f"SSE event data: {json.dumps(event_data, indent=2)}")
                        
                        # Look for result in the event data
                        if 'result' in event_data:
                            result_data = event_data['result']
                            break
                        elif 'error' in event_data:
                            raise Exception(f"MCP Error: {event_data['error']}")
                            
                    except json.JSONDecodeError as e:
                        self._debug_log(f"JSON decode error: {e}, data: {data}")
                        continue
                        
                # Safety check to avoid infinite loops
                if line_count > 1000:
                    self._debug_log("Too many lines in SSE stream, breaking")
                    break
                    
            return result_data
            
        except Exception as e:
            self._debug_log(f"Error reading SSE response: {str(e)}")
            raise Exception(f"Error reading SSE response: {str(e)}")
    
    def _format_response(self, result: dict) -> str:
        """Format the response from MCP server with better handling"""
        if "error" in result:
            error = result["error"]
            if isinstance(error, dict):
                return f"❌ Error: {error.get('message', str(error))}"
            return f"❌ Error: {error}"
        
        if "result" in result:
            data = result["result"]
            
            # Handle None or empty results
            if data is None:
                return "ℹ️ No data returned from the server"
            
            if isinstance(data, dict):
                if "content" in data:
                    # Handle content response
                    content = data["content"]
                    if isinstance(content, list) and len(content) > 0:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            return first_content.get("text", str(content))
                        return str(first_content)
                    else:
                        return str(content) if content else "ℹ️ Empty content"
                        
                elif "emails" in data:
                    # Handle Gmail emails response
                    emails = data["emails"]
                    if isinstance(emails, list) and len(emails) > 0:
                        email_list = []
                        for i, email in enumerate(emails[:10]):  # Show first 10
                            sender = email.get("from", "Unknown")
                            subject = email.get("subject", "No subject")
                            date = email.get("date", "Unknown date")
                            snippet = email.get("snippet", email.get("body", ""))[:100]
                            email_list.append(f"{i+1}. From: {sender}\n   Subject: {subject}\n   Date: {date}\n   Preview: {snippet}...")
                        return f"📧 Found {len(emails)} emails:\n\n" + "\n\n".join(email_list)
                    else:
                        return "📧 No emails found"
                        
                elif "repositories" in data or "repos" in data:
                    # Handle GitHub repositories response
                    repos = data.get("repositories", data.get("repos", []))
                    if isinstance(repos, list) and len(repos) > 0:
                        repo_list = []
                        for i, repo in enumerate(repos[:10]):  # Show first 10
                            name = repo.get("name", "Unknown")
                            description = repo.get("description", "No description")
                            updated = repo.get("updated_at", "Unknown")
                            repo_list.append(f"{i+1}. {name}\n   Description: {description}\n   Updated: {updated}")
                        return f"🐙 Found {len(repos)} repositories:\n\n" + "\n\n".join(repo_list)
                    else:
                        return "🐙 No repositories found"
                        
                else:
                    # Handle other dict responses
                    formatted_lines = []
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            formatted_lines.append(f"**{key}**: {json.dumps(value, indent=2)}")
                        else:
                            formatted_lines.append(f"**{key}**: {value}")
                    return "\n".join(formatted_lines) if formatted_lines else "ℹ️ Empty response"
                    
            elif isinstance(data, list):
                if len(data) > 0:
                    return "\n".join([f"• {item}" for item in data])
                else:
                    return "ℹ️ Empty list returned"
            else:
                return str(data) if data else "ℹ️ Empty response"
        
        return "✅ Operation completed successfully"

# ============================================================================
# LANGCHAIN INTEGRATION SECTION - FIXED VERSION
# ============================================================================

class MCPServerTool(BaseTool):
    """FIXED LangChain Tool for MCP Server Integration - No more event loop errors!"""
    name: str = Field()
    description: str = Field()
    server_adapter: Any = Field()
    tool_name: str = Field()
    
    def _run(self, query: str) -> str:
        """Execute the MCP tool with the given query - FIXED VERSION"""
        try:
            # Parse parameters from query
            params = self._parse_query_params(query)
            
            # Use a completely separate thread with its own event loop
            result = self._run_async_in_thread(params)
            return result
            
        except Exception as e:
            error_trace = traceback.format_exc()
            return f"Error executing {self.tool_name}: {str(e)}\n\nFull trace:\n{error_trace}"
    
    def _run_async_in_thread(self, params: dict) -> str:
        """Run async code in a dedicated thread with proper event loop management"""
        def run_in_new_thread():
            # Create a completely fresh event loop for this thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            
            try:
                # Run the actual async operation
                result = new_loop.run_until_complete(
                    self.server_adapter.execute_tool(self.tool_name, params)
                )
                return result
            except Exception as e:
                return f"Async execution error: {str(e)}"
            finally:
                # Always clean up the loop
                try:
                    new_loop.close()
                except:
                    pass
        
        # Execute in a separate thread with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_new_thread)
            try:
                result = future.result(timeout=90)  # Increased timeout to 90s
                return result
            except concurrent.futures.TimeoutError:
                return f"⏱️ Timeout executing {self.tool_name} (90s limit exceeded)"
            except Exception as e:
                return f"Thread execution error: {str(e)}"
    
    def _parse_query_params(self, query: str) -> dict:
        """Parse query into parameters for the tool"""
        params = {}
        
        if self.tool_name == "GMAIL_SEND_EMAIL":
            params = {
                "to": "user@example.com",
                "subject": f"Message: {query}",
                "body": query
            }
        elif self.tool_name == "GMAIL_GET_MESSAGES" or self.tool_name == "GMAIL_SEARCH_MESSAGES":
            # Parse email queries for better parameters
            query_lower = query.lower()
            params = {}
            
            if "yesterday" in query_lower:
                # Add date filtering for yesterday
                yesterday = datetime.now() - timedelta(days=1)
                params["query"] = f"after:{yesterday.strftime('%Y/%m/%d')} before:{(yesterday + timedelta(days=1)).strftime('%Y/%m/%d')}"
                params["max_results"] = 20
            elif "today" in query_lower:
                today = datetime.now()
                params["query"] = f"after:{today.strftime('%Y/%m/%d')}"
                params["max_results"] = 20
            elif "week" in query_lower:
                week_ago = datetime.now() - timedelta(days=7)
                params["query"] = f"after:{week_ago.strftime('%Y/%m/%d')}"
                params["max_results"] = 50
            else:
                params["query"] = "in:inbox"
                params["max_results"] = 20
                
        elif self.tool_name == "connect-gmail":
            # Connection tool doesn't need parameters
            params = {}
        elif self.tool_name == "GITHUB_LIST_REPOS":
            # For GitHub repo listing
            params = {
                "type": "all",
                "sort": "updated",
                "direction": "desc"
            }
        elif self.tool_name == "GITHUB_LIST_COMMITS":
            # For GitHub commits
            params = {
                "since": (datetime.now() - timedelta(days=7)).isoformat(),
                "per_page": 10
            }
        else:
            params = {"query": query}
        
        return params

class LangChainWorkflowAssistant:
    """
    Enhanced Workflow Assistant with LangChain integration - IMPROVED VERSION
    Provides intelligent processing, conversation memory, and multi-tool coordination.
    """
    
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
        
        self.active_servers = {}  # server_name -> MCPServerAdapter
        self.api_key = api_key
        self.model = model
        
        # Initialize LangChain components
        self._initialize_langchain()
    
    def _initialize_langchain(self):
        """Initialize LangChain components"""
        # Initialize tools list first
        self.langchain_tools = []
        
        if not self.api_key:
            self.llm = None
            self.memory = None
            self.agent = None
            return
        
        try:
            # Initialize LLM with OpenRouter
            self.llm = ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                openai_api_base="https://openrouter.ai/api/v1",
                temperature=0.7,
                max_tokens=2000
            )
            
            # Initialize memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create workflow analysis chain
            workflow_prompt = PromptTemplate(
                input_variables=["user_input"],
                template="""
                You are a helpful workflow assistant that can use various tools to help users.
                
                User request: {user_input}
                
                Analyze this request and determine:
                1. What tools should be used
                2. What parameters are needed
                3. The order of operations
                
                Provide a clear explanation of your reasoning and the planned actions.
                """
            )
            
            self.workflow_chain = LLMChain(
                llm=self.llm,
                prompt=workflow_prompt,
                memory=self.memory
            )
            
            # Update agent with current tools
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
    
    async def add_server(self, server_name: str, server_url: str):
        """Add and connect to an MCP server"""
        if server_name in self.server_templates:
            # Create server info with URL
            server_info = MCPServerInfo(
                name=self.server_templates[server_name].name,
                description=self.server_templates[server_name].description,
                capabilities=self.server_templates[server_name].capabilities,
                icon=self.server_templates[server_name].icon,
                category=self.server_templates[server_name].category,
                url=server_url
            )
            
            # Create adapter and connect
            adapter = MCPServerAdapter(server_info)
            await adapter.connect()
            
            # Store active server
            self.active_servers[server_name] = adapter
            
            # Add LangChain tools for this server
            self._add_langchain_tools_for_server(server_name, adapter)
            
            return True
        return False
    
    def _add_langchain_tools_for_server(self, server_name: str, adapter: MCPServerAdapter):
        """Add LangChain tools for a connected MCP server"""
        for capability in adapter.server_info.capabilities:
            tool = MCPServerTool(
                name=f"{server_name}_{capability}",
                description=f"{adapter.server_info.description} - {capability}",
                server_adapter=adapter,
                tool_name=capability
            )
            self.langchain_tools.append(tool)
        
        # Update agent with new tools
        self._update_agent()
    
    async def remove_server(self, server_name: str):
        """Remove and disconnect from an MCP server"""
        if server_name in self.active_servers:
            await self.active_servers[server_name].disconnect()
            del self.active_servers[server_name]
            
            # Remove related LangChain tools
            self.langchain_tools = [
                tool for tool in self.langchain_tools 
                if not tool.name.startswith(f"{server_name}_")
            ]
            
            # Update agent
            self._update_agent()
    
    async def process_request(self, user_input: str):
        """Process user request with improved error handling and fallbacks"""
        if not self.active_servers:
            yield "❌ No MCP servers connected. Please add server URLs in the sidebar."
            return
        
        yield f"🧠 Analyzing request: {user_input}"
        await asyncio.sleep(0.5)
        
        try:
            # First, try direct execution (more reliable)
            if await self._try_direct_execution(user_input):
                async for response in self._direct_process_request(user_input):
                    yield response
                return
            
            # If direct execution isn't applicable, try LangChain
            if self.api_key and self.agent:
                yield f"🚀 Using LangChain agent..."
                
                try:
                    # Execute with LangChain agent in a safe way
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        self._run_agent_safely, 
                        user_input
                    )
                    
                    if result and "error" not in result.lower():
                        yield f"✅ LangChain Result:\n{result}"
                        return
                    else:
                        yield f"⚠️ LangChain issue: {result}"
                        yield f"🔄 Trying direct execution..."
                
                except Exception as e:
                    yield f"⚠️ LangChain error: {str(e)}"
                    yield f"🔄 Falling back to direct execution..."
            
            # Fallback to direct execution
            async for response in self._direct_process_request(user_input):
                yield response
                
        except Exception as e:
            yield f"❌ Error processing request: {str(e)}"
            yield f"🔄 Trying simple fallback..."
            
            # Final fallback
            async for response in self._simple_process_request(user_input):
                yield response
    
    async def _try_direct_execution(self, user_input: str) -> bool:
        """Check if we should try direct execution first"""
        user_input_lower = user_input.lower()
        
        # Direct execution for simple, clear requests
        direct_keywords = [
            "get emails", "check emails", "list repos", "show repos",
            "github commits", "send email", "yesterday emails"
        ]
        
        return any(keyword in user_input_lower for keyword in direct_keywords)
    
    def _run_agent_safely(self, user_input: str) -> str:
        """Run LangChain agent in a safe way that won't conflict with event loops"""
        try:
            # Simple synchronous execution
            result = self.agent.run(user_input)
            return str(result)
        except Exception as e:
            return f"Agent execution error: {str(e)}"
    
    async def _direct_process_request(self, user_input: str):
        """Direct execution without LangChain agent - more reliable"""
        # Parse intent
        intent = self._parse_intent(user_input)
        
        yield f"🎯 Direct execution mode activated"
        await asyncio.sleep(0.5)
        
        # Execute based on intent
        server_name = intent.get("server")
        if server_name and server_name in self.active_servers:
            yield f"🔄 {server_name}: Processing request..."
            
            adapter = self.active_servers[server_name]
            
            try:
                # Special handling for Gmail - ensure connection first
                if server_name == "gmail" and intent["action"] != "connect-gmail":
                    yield f"📧 Gmail: Ensuring connection..."
                    
                    # Try to connect first if not already done
                    try:
                        connect_result = await adapter.execute_tool("connect-gmail", {})
                        yield f"🔗 Gmail Connection Status: {connect_result[:100]}..."
                        
                        # Wait a moment for connection to establish
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        yield f"🔗 Connection attempt: {str(e)}"
                
                # Execute the main action
                result = await adapter.execute_tool(
                    intent["action"], 
                    intent.get("params", {})
                )
                
                # Format and yield the result
                yield f"{adapter.server_info.icon} **{adapter.server_info.name} Results:**"
                yield f"{result}"
                
            except Exception as e:
                yield f"❌ {server_name} Error: {str(e)}"
                
        elif intent["type"] == "complex_workflow":
            async for response in self._handle_complex_workflow(user_input):
                yield response
        else:
            yield "ℹ️ I understand you want to work with external services. Please make sure the relevant servers are connected in the sidebar."
    
    async def _simple_process_request(self, user_input: str):
        """Simplest fallback processing"""
        intent = self._parse_intent(user_input)
        
        yield f"🔧 Simple processing mode"
        
        # List available servers and capabilities
        if self.active_servers:
            available = []
            for name, adapter in self.active_servers.items():
                available.append(f"{adapter.server_info.icon} {adapter.server_info.name}")
            
            yield f"📋 Available services: {', '.join(available)}"
            yield f"💡 Try asking me to 'check Gmail emails', 'list GitHub repos', or 'get yesterday's emails'"
        else:
            yield "❌ No servers connected. Please add server URLs in the sidebar first."
    
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
                action = "GITHUB_LIST_REPOS"  # Default to repo listing
            return {"type": "github", "server": "github", "action": action, "params": {}}
        
        # Gmail operations
        elif any(word in user_input_lower for word in ["gmail", "email", "mail"]):
            if "send" in user_input_lower:
                action = "GMAIL_SEND_EMAIL"
                params = {"to": "user@example.com", "subject": "Test", "body": "Hello!"}
            elif "yesterday" in user_input_lower:
                action = "GMAIL_SEARCH_MESSAGES"
                yesterday = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{(yesterday + timedelta(days=1)).strftime('%Y/%m/%d')}",
                    "max_results": 20
                }
            else:
                action = "GMAIL_SEARCH_MESSAGES"
                params = {"query": "in:inbox", "max_results": 20}
                
            return {"type": "gmail", "server": "gmail", "action": action, "params": params}
        
        # Complex workflows
        elif any(phrase in user_input_lower for phrase in ["workflow", "automate", "summary", "report"]):
            return {"type": "complex_workflow"}
        
        return {"type": "general"}
    
    async def _handle_complex_workflow(self, user_input: str):
        """Handle complex workflows across multiple servers"""
        yield "🔄 Executing multi-service workflow..."
        
        # Example: GitHub activity -> Gmail notification
        if "github" in self.active_servers and "gmail" in self.active_servers:
            # Step 1: Get GitHub data
            yield "📋 Step 1: Fetching GitHub activity..."
            github_adapter = self.active_servers["github"]
            try:
                github_result = await github_adapter.execute_tool("GITHUB_LIST_REPOS", {})
                yield f"🐙 GitHub: {github_result[:200]}..."
                
                # Step 2: Send to Gmail
                yield "📋 Step 2: Preparing summary email..."
                gmail_adapter = self.active_servers["gmail"]
                summary = f"📊 GitHub Activity Summary:\n{github_result[:500]}..."
                
                gmail_result = await gmail_adapter.execute_tool("GMAIL_SEND_EMAIL", {
                    "to": "yourself@example.com",
                    "subject": "GitHub Activity Report",
                    "body": summary
                })
                yield f"📧 Gmail: {gmail_result}"
                
            except Exception as e:
                yield f"❌ Workflow error: {str(e)}"
        
        yield "✅ Workflow completed!"

# ============================================================================
# USER INTERFACE COMPONENTS SECTION
# ============================================================================

class UIComponents:
    """Reusable UI components for the Streamlit interface"""
    
    @staticmethod
    def render_header():
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Personal Workflow Assistant</h1>
            <p>Connect your Composio MCP Servers via HTTPS Streams with LangChain Intelligence</p>
            <p style="font-size: 0.9em; opacity: 0.8;">✅ Fixed Version - Event Loop Issues Resolved</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_server_configuration(assistant):
        """Render server configuration in sidebar"""
        st.subheader("🔌 MCP Servers")
        st.info("💡 Add your Composio MCP server HTTPS stream URLs below")
        
        # Group by categories
        categories = {}
        for server_name, template in assistant.server_templates.items():
            if template.category not in categories:
                categories[template.category] = []
            categories[template.category].append((server_name, template))
        
        # Display server configuration by category
        for category, servers in categories.items():
            st.write(f"**{category}**")
            
            for server_name, template in servers:
                with st.expander(f"{template.icon} {template.name}"):
                    st.write(f"*{template.description}*")
                    st.write(f"**Capabilities**: {', '.join(template.capabilities[:3])}...")
                    
                    # URL input for this server
                    current_url = st.session_state.server_urls.get(server_name, "")
                    new_url = st.text_input(
                        "MCP Server URL",
                        value=current_url,
                        key=f"url_{server_name}",
                        placeholder="https://mcp.composio.dev/composio/server/...",
                        help="Paste your unique Composio MCP server URL here"
                    )
                    
                    # Update URL in session state
                    if new_url != current_url:
                        st.session_state.server_urls[server_name] = new_url
                    
                    # Connection controls
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if server_name not in assistant.active_servers:
                            if st.button(f"Connect", key=f"connect_{server_name}", disabled=not new_url):
                                UIComponents._handle_server_connection(assistant, server_name, new_url, template)
                        else:
                            st.success("✅ Connected")
                    
                    with col2:
                        if server_name in assistant.active_servers:
                            if st.button(f"Disconnect", key=f"disconnect_{server_name}"):
                                UIComponents._handle_server_disconnection(assistant, server_name, template)
    
    @staticmethod
    def _handle_server_connection(assistant, server_name, new_url, template):
        """Handle server connection"""
        if new_url:
            try:
                # Show connection status
                with st.spinner(f"Connecting to {template.name}..."):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        success = loop.run_until_complete(assistant.add_server(server_name, new_url))
                        if success:
                            st.success(f"✅ Connected to {template.name}")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to connect to {template.name}")
                    finally:
                        loop.close()
                        
            except Exception as e:
                st.error(f"❌ Connection failed: {str(e)}")
                
                # Show debug info in expander
                with st.expander("🔧 Debug Information"):
                    st.code(traceback.format_exc())
    
    @staticmethod
    def _handle_server_disconnection(assistant, server_name, template):
        """Handle server disconnection"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(assistant.remove_server(server_name))
            st.success(f"Disconnected from {template.name}")
            st.rerun()
        except Exception as e:
            st.error(f"Error disconnecting: {str(e)}")
        finally:
            loop.close()
    
    @staticmethod
    def render_api_configuration(assistant):
        """Render API configuration section"""
        st.header("⚙️ Configuration")
        
        # OpenRouter API Key (for LLM processing)
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="For LangChain LLM processing (get one at openrouter.ai)",
            placeholder="sk-or-..."
        )
        
        # Model selection
        model_options = {
            "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet",
            "Claude 3.5 Haiku": "anthropic/claude-3.5-haiku", 
            "GPT-4o": "openai/gpt-4o",
            "GPT-4o Mini": "openai/gpt-4o-mini",
            "Gemini Pro 1.5": "google/gemini-pro-1.5"
        }
        
        selected_model_name = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        
        selected_model = model_options[selected_model_name]
        
        # Update assistant with API key and model
        if api_key and (not hasattr(st.session_state.assistant, 'api_key') or 
                       st.session_state.assistant.api_key != api_key or
                       st.session_state.assistant.model != selected_model):
            st.session_state.assistant = create_fixed_assistant(api_key, selected_model)
        
        # LangChain Status
        if api_key:
            st.success("🔗 LangChain: Ready")
            if hasattr(st.session_state.assistant, 'agent') and st.session_state.assistant.agent:
                st.info(f"🤖 Agent: {len(st.session_state.assistant.langchain_tools)} tools loaded")
            else:
                st.info("🤖 Agent: Direct execution mode (more reliable)")
        else:
            st.warning("🔗 LangChain: API key required for advanced features")
            st.info("💡 You can still use direct execution without an API key!")
        
        return api_key, selected_model
    
    @staticmethod
    def render_chat_interface(assistant):
        """Render the main chat interface"""
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
                    elif "🔧" in content and "DEBUG" in content:
                        # Show debug information in a code block
                        with st.expander("🔧 Debug Information", expanded=False):
                            st.code(content, language="text")
                    else:
                        st.markdown(content)
                else:
                    st.markdown(message["content"])
        
        # Chat input
        if assistant.active_servers:
            if prompt := st.chat_input("What would you like me to help you with?"):
                UIComponents._process_chat_message(assistant, prompt)
        else:
            st.info("👈 Please connect to MCP servers in the sidebar to start chatting.")
    
    @staticmethod
    def _process_chat_message(assistant, prompt):
        """Process a chat message"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with assistant
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            async def process_message():
                response_parts = []
                try:
                    async for chunk in assistant.process_request(prompt):
                        response_parts.append(chunk)
                        current_response = "\n".join(response_parts)
                        
                        if "🧠" in chunk or "🚀" in chunk:
                            message_placeholder.markdown(f'<div class="langchain-response">{current_response}</div>', unsafe_allow_html=True)
                        elif "🔄" in chunk or "📧" in chunk or "🐙" in chunk:
                            message_placeholder.markdown(f'<div class="tool-response">{current_response}</div>', unsafe_allow_html=True)
                        else:
                            message_placeholder.markdown(current_response)
                            
                except Exception as e:
                    error_msg = f"❌ Error processing message: {str(e)}"
                    response_parts.append(error_msg)
                    message_placeholder.error(error_msg)
                
                return "\n".join(response_parts)
            
            # Run async processing
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                full_response = loop.run_until_complete(process_message())
            except Exception as e:
                full_response = f"Error: {str(e)}"
                message_placeholder.error(full_response)
            finally:
                loop.close()
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()
    
    @staticmethod
    def render_sidebar_info(assistant):
        """Render information panels in sidebar"""
        st.divider()
        
        # Connection Status
        st.subheader("📊 Active Connections")
        if assistant.active_servers:
            for server_name, adapter in assistant.active_servers.items():
                st.markdown(f"✅ {adapter.server_info.icon} **{adapter.server_info.name}**")
        else:
            st.info("No servers connected")
        
        st.divider()
        
        # Quick Actions
        st.subheader("💡 Quick Actions")
        
        if assistant.active_servers:
            quick_actions = {}
            
            for server_name, adapter in assistant.active_servers.items():
                server_info = adapter.server_info
                if server_name == "github":
                    quick_actions["🐙 List Repositories"] = "Show me my GitHub repositories"
                    quick_actions["🔍 Recent Commits"] = "Show my latest GitHub commits"
                elif server_name == "gmail":
                    quick_actions["📧 Check Inbox"] = "Get my recent emails"
                    quick_actions["📮 Yesterday's Emails"] = "Get my emails from yesterday"
            
            if len(assistant.active_servers) >= 2:
                quick_actions["🔄 GitHub → Email"] = "Send me a summary of my GitHub activity via email"
            
            for action_name, action_prompt in quick_actions.items():
                if st.button(action_name, key=f"quick_{action_name}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": action_prompt})
                    st.rerun()
        else:
            st.info("Connect servers to see quick actions")
    
    @staticmethod
    def render_tools_panel(assistant):
        """Render tools information panel"""
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
    
    @staticmethod
    def render_status_panel(assistant):
        """Render system status panel"""
        st.subheader("🔗 System Status")
        
        # LangChain Status
        if hasattr(assistant, 'api_key') and assistant.api_key:
            st.success("✅ API Key: Connected")
            st.info(f"🤖 Model: {assistant.model}")
            
            if hasattr(assistant, 'agent') and assistant.agent:
                st.success(f"🛠️ Agent: Active ({len(assistant.langchain_tools)} tools)")
            else:
                st.info("🎯 Mode: Direct execution (recommended)")
                
        else:
            st.info("💡 Running in direct execution mode")
            st.write("This is actually more reliable for simple tasks!")
        
        # Connection Health
        if assistant.active_servers:
            healthy = sum(1 for adapter in assistant.active_servers.values() if adapter.connected)
            total = len(assistant.active_servers)
            st.metric("Server Health", f"{healthy}/{total}")
        
    @staticmethod
    def render_instructions():
        """Render instructions panel"""
        st.subheader("📖 How to Use")
        st.markdown("""
        **Quick Start:**
        1. **Get MCP URLs**: Create servers at [Composio MCP](https://mcp.composio.dev/) and copy URLs
        2. **Connect Servers**: Add URLs in the sidebar and click Connect
        3. **Start Chatting**: Use natural language to interact with your services
        
        **Example Commands:**
        - "Get my emails from yesterday"
        - "List my GitHub repositories"
        - "Show recent commits"
        - "Send me a GitHub activity summary"
        
        **✅ Fixed Issues:**
        - Event loop conflicts resolved
        - Better error handling
        - More reliable connections
        - Improved timeout handling
        """)
    
    @staticmethod
    def render_examples():
        """Render example prompts"""
        st.subheader("💡 Try These Examples")
        example_prompts = [
            "📧 Get my emails from yesterday",
            "🐙 Show me my GitHub repositories", 
            "🔍 List recent commits",
            "📊 Send me a GitHub activity summary via email",
            "📮 Check my inbox for new messages"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, key=f"example_{hash(prompt)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

# ============================================================================
# SESSION STATE MANAGEMENT SECTION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = create_fixed_assistant()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'server_urls' not in st.session_state:
        st.session_state.server_urls = {}

def create_fixed_assistant(api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
    """Create an assistant with the fixed event loop handling"""
    return LangChainWorkflowAssistant(api_key, model)

# ============================================================================
# MAIN APPLICATION SECTION
# ============================================================================

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    UIComponents.render_header()
    
    # Sidebar Configuration
    with st.sidebar:
        # API Configuration
        api_key, selected_model = UIComponents.render_api_configuration(st.session_state.assistant)
        
        st.divider()
        
        # MCP Server Configuration
        UIComponents.render_server_configuration(st.session_state.assistant)
        
        # Sidebar info panels
        UIComponents.render_sidebar_info(st.session_state.assistant)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        UIComponents.render_chat_interface(st.session_state.assistant)
    
    with col2:
        # Tools panel
        UIComponents.render_tools_panel(st.session_state.assistant)
        
        st.divider()
        
        # Status panel
        UIComponents.render_status_panel(st.session_state.assistant)
        
        st.divider()
        
        # Instructions
        UIComponents.render_instructions()
        
        st.divider()
        
        # Example prompts
        UIComponents.render_examples()
    
    # Footer section
    st.divider()
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Connected Servers", len(st.session_state.assistant.active_servers))
    
    with col2:
        tools_count = len(getattr(st.session_state.assistant, 'langchain_tools', []))
        st.metric("Available Tools", tools_count)
    
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if hasattr(st.session_state.assistant, 'memory') and st.session_state.assistant.memory:
                st.session_state.assistant.memory.clear()
            st.rerun()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
    
    @staticmethod
    def render_examples():
        """Render example prompts"""
        st.subheader("💡 Example Prompts")
        example_prompts = [
            "📊 Create a summary of my GitHub activity and send it via email",
            "📧 Get my emails from yesterday",
            "🐙 Show me all my GitHub repositories",
            "🔄 Set up an automated workflow for daily GitHub activity reports",
            "📧 Send an email with my latest GitHub commits"
        ]
        
        for prompt in example_prompts:
            if st.button(prompt, key=f"example_{hash(prompt)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()

# ============================================================================
# SESSION STATE MANAGEMENT SECTION
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'assistant' not in st.session_state:
        st.session_state.assistant = LangChainWorkflowAssistant()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'server_urls' not in st.session_state:
        st.session_state.server_urls = {}

# ============================================================================
# MAIN APPLICATION SECTION
# ============================================================================

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Render header
    UIComponents.render_header()
    
    # Sidebar Configuration
    with st.sidebar:
        # API Configuration
        api_key, selected_model = UIComponents.render_api_configuration(st.session_state.assistant)
        
        st.divider()
        
        # MCP Server Configuration
        UIComponents.render_server_configuration(st.session_state.assistant)
        
        # Sidebar info panels
        UIComponents.render_sidebar_info(st.session_state.assistant)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        UIComponents.render_chat_interface(st.session_state.assistant)
    
    with col2:
        # Tools panel
        UIComponents.render_tools_panel(st.session_state.assistant)
        
        st.divider()
        
        # Status panel
        UIComponents.render_status_panel(st.session_state.assistant)
        
        st.divider()
        
        # Instructions
        UIComponents.render_instructions()
        
        st.divider()
        
        # Example prompts
        UIComponents.render_examples()
    
    # Footer section
    st.divider()
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Connected Servers", len(st.session_state.assistant.active_servers))
    
    with col2:
        st.metric("LangChain Tools", len(getattr(st.session_state.assistant, 'langchain_tools', [])))
    
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.assistant.memory:
                st.session_state.assistant.memory.clear()
            st.rerun()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
