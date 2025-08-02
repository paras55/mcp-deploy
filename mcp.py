import streamlit as st
import asyncio
import json
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import os
import time
import aiohttp

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
</style>
""", unsafe_allow_html=True)

# Configuration classes
@dataclass
class MCPServerInfo:
    name: str
    description: str
    capabilities: List[str]
    icon: str
    category: str
    url: str = ""
    connected: bool = False

# LangChain Tool for MCP Server Integration
class MCPServerTool(BaseTool):
    name: str = Field()
    description: str = Field()
    server_adapter: Any = Field()
    tool_name: str = Field()
    
    def _run(self, query: str) -> str:
        """Execute the MCP tool with the given query"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Parse parameters from query if needed
            params = self._parse_query_params(query)
            
            result = loop.run_until_complete(
                self.server_adapter.execute_tool(self.tool_name, params)
            )
            return result
        except Exception as e:
            return f"Error executing {self.tool_name}: {str(e)}"
        finally:
            if 'loop' in locals():
                loop.close()
    
    def _parse_query_params(self, query: str) -> dict:
        """Parse query into parameters for the tool"""
        # Simple parameter parsing - can be enhanced
        params = {}
        
        if self.tool_name == "send_email":
            params = {
                "subject": f"Message: {query}",
                "body": query
            }
        else:
            params = {"query": query}
        
        return params

# Real MCP Adapter for individual servers
class MCPServerAdapter:
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.session = None
        self.connected = False
    
    async def connect(self):
        """Connect to the MCP server via HTTPS stream"""
        if not self.server_info.url:
            raise Exception("No server URL provided")
            
        try:
            # Initialize HTTP session with proper headers for MCP
            self.session = aiohttp.ClientSession(
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "User-Agent": "Workflow-Assistant/1.0.0"
                },
                timeout=aiohttp.ClientTimeout(total=30)
            )
            
            # Test connection with initialize request
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
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
            
            async with self.session.post(
                self.server_info.url, 
                json=init_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result:
                        self.connected = True
                        self.server_info.connected = True
                        return True
                    else:
                        raise Exception(f"Initialize failed: {result.get('error', 'Unknown error')}")
                else:
                    error_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_text}")
                    
        except Exception as e:
            if self.session:
                await self.session.close()
            raise Exception(f"Connection failed: {str(e)}")
    
    async def disconnect(self):
        """Disconnect from the server"""
        self.connected = False
        self.server_info.connected = False
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
                "id": 2,
                "method": "tools/list"
            }
            
            async with self.session.post(
                self.server_info.url, 
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if "result" in result:
                        return result["result"].get("tools", [])
                    else:
                        return []
                else:
                    return []
        except:
            return []
    
    async def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute a tool via the MCP server"""
        if not self.connected or not self.session:
            raise Exception("Not connected to server")
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": parameters
                }
            }
            
            async with self.session.post(
                self.server_info.url, 
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return self._format_response(result)
                else:
                    error_text = await response.text()
                    return f"❌ Error executing {tool_name}: {error_text}"
                    
        except asyncio.TimeoutError:
            return f"⏱️ Timeout executing {tool_name}"
        except Exception as e:
            return f"❌ Error executing {tool_name}: {str(e)}"
    
    def _format_response(self, result: dict) -> str:
        """Format the response from MCP server"""
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        if "result" in result:
            data = result["result"]
            if isinstance(data, dict):
                if "content" in data:
                    # Handle content response
                    content = data["content"]
                    if isinstance(content, list) and len(content) > 0:
                        return content[0].get("text", str(content))
                    else:
                        return str(content)
                else:
                    # Handle other dict responses
                    formatted_lines = []
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            formatted_lines.append(f"**{key}**: {json.dumps(value, indent=2)}")
                        else:
                            formatted_lines.append(f"**{key}**: {value}")
                    return "\n".join(formatted_lines)
            elif isinstance(data, list):
                return "\n".join([f"• {item}" for item in data])
            else:
                return str(data)
        
        return "✅ Operation completed successfully"

# Enhanced Workflow Assistant with LangChain
class LangChainWorkflowAssistant:
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
        # Available MCP server templates
        self.server_templates = {
            "github": MCPServerInfo(
                name="GitHub",
                description="GitHub repository management and operations",
                capabilities=["get_repo", "create_issue", "get_commits", "search_code"],
                icon="🐙",
                category="Development"
            ),
            "gmail": MCPServerInfo(
                name="Gmail",
                description="Gmail email management",
                capabilities=["send_email", "get_emails", "create_draft"],
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
        
        # Initialize LLM with OpenRouter
        self.llm = ChatOpenAI(
            model_name=self.model.split('/')[-1],  # Extract model name
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
            input_variables=["user_input", "available_tools"],
            template="""
            You are a helpful workflow assistant that can use various tools to help users.
            
            Available tools: {available_tools}
            
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
    
    def _update_agent(self):
        """Update the LangChain agent with current tools"""
        if not self.llm:
            return
        
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
        """Process user request using LangChain agent"""
        if not self.active_servers:
            yield "❌ No MCP servers connected. Please add server URLs in the sidebar."
            return
        
        if not self.api_key:
            yield "❌ No OpenRouter API key provided. Please add your API key in the sidebar."
            return
        
        if not self.agent:
            yield "❌ LangChain agent not initialized. Please check your API key and try again."
            return
        
        yield f"🧠 LangChain: Analyzing request..."
        await asyncio.sleep(0.5)
        
        try:
            # Get available tools description
            available_tools = [
                f"{tool.name}: {tool.description}" 
                for tool in self.langchain_tools
            ]
            
            # First, analyze the workflow
            if self.workflow_chain:
                yield f"🔍 LangChain: Planning workflow..."
                workflow_analysis = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.workflow_chain.run({
                        "user_input": user_input,
                        "available_tools": "\n".join(available_tools)
                    })
                )
                yield f"📋 Workflow Plan:\n{workflow_analysis}"
            
            # Execute with agent
            yield f"🚀 LangChain: Executing with agent..."
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.agent.run(user_input)
            )
            
            yield f"✅ LangChain Result:\n{result}"
            
        except Exception as e:
            yield f"❌ LangChain Error: {str(e)}"
            
            # Fallback to simple processing
            yield f"🔄 Falling back to simple processing..."
            async for response in self._simple_process_request(user_input):
                yield response
    
    async def _simple_process_request(self, user_input: str):
        """Fallback simple processing without LangChain"""
        # Parse user intent
        intent = self._parse_intent(user_input)
        
        yield f"🧠 Understanding request: {user_input}"
        await asyncio.sleep(0.5)
        
        # Execute based on intent
        server_name = intent.get("server")
        if server_name and server_name in self.active_servers:
            yield f"🔄 {server_name}: Processing request..."
            
            adapter = self.active_servers[server_name]
            result = await adapter.execute_tool(intent["action"], intent.get("params", {}))
            yield f"{adapter.server_info.icon} {adapter.server_info.name}: {result}"
        
        elif intent["type"] == "complex_workflow":
            async for response in self._handle_complex_workflow(user_input):
                yield response
        else:
            # List available servers and capabilities
            available = []
            for name, adapter in self.active_servers.items():
                available.append(f"{adapter.server_info.icon} {adapter.server_info.name}")
            
            yield f"I can help you with: {', '.join(available)}. Try asking me to check GitHub commits, send a Slack message, create a calendar event, or run a workflow across multiple services."
    
    def _parse_intent(self, user_input: str) -> dict:
        """Parse user intent to determine which server and action to use"""
        user_input_lower = user_input.lower()
        
        # GitHub operations
        if any(word in user_input_lower for word in ["github", "commit", "repo", "issue"]):
            action = "get_commits" if "commit" in user_input_lower else "get_repo"
            return {"type": "github", "server": "github", "action": action, "params": {}}
        
        # Gmail operations
        elif any(word in user_input_lower for word in ["gmail", "email", "mail"]):
            action = "send_email" if "send" in user_input_lower else "get_emails"
            params = {"to": "", "subject": "Test", "body": "Hello!"} if action == "send_email" else {}
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
            github_result = await github_adapter.execute_tool("get_commits", {})
            yield f"🐙 GitHub: {github_result[:200]}..."
            
            # Step 2: Send to Gmail
            yield "📋 Step 2: Sending summary via email..."
            gmail_adapter = self.active_servers["gmail"]
            summary = f"📊 GitHub Activity Summary: {github_result[:100]}..."
            gmail_result = await gmail_adapter.execute_tool("send_email", {
                "to": "yourself@example.com",
                "subject": "GitHub Activity Report",
                "body": summary
            })
            yield f"📧 Gmail: {gmail_result}"
        
        yield "✅ Workflow completed successfully!"

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = LangChainWorkflowAssistant()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'server_urls' not in st.session_state:
    st.session_state.server_urls = {}

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Personal Workflow Assistant</h1>
        <p>Connect your Composio MCP Servers via HTTPS Streams with LangChain Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
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
            st.session_state.assistant = LangChainWorkflowAssistant(api_key, selected_model)
        
        # LangChain Status
        if api_key:
            st.success("🔗 LangChain: Ready")
            if st.session_state.assistant.agent:
                st.info(f"🤖 Agent: {len(st.session_state.assistant.langchain_tools)} tools loaded")
        else:
            st.warning("🔗 LangChain: API key required")
        
        st.divider()
        
        # MCP Server Configuration
        st.subheader("🔌 MCP Servers")
        st.info("💡 Add your Composio MCP server HTTPS stream URLs below")
        
        assistant = st.session_state.assistant
        
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
                                if new_url:
                                    try:
                                        # Show connection status
                                        with st.spinner(f"Connecting to {template.name}..."):
                                            loop = asyncio.new_event_loop()
                                            asyncio.set_event_loop(loop)
                                            success = loop.run_until_complete(assistant.add_server(server_name, new_url))
                                            loop.close()
                                            
                                            if success:
                                                st.success(f"✅ Connected to {template.name}")
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Failed to connect to {template.name}")
                                    except Exception as e:
                                        st.error(f"❌ Connection failed: {str(e)}")
                        else:
                            st.success("✅ Connected")
                    
                    with col2:
                        if server_name in assistant.active_servers:
                            if st.button(f"Disconnect", key=f"disconnect_{server_name}"):
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                try:
                                    loop.run_until_complete(assistant.remove_server(server_name))
                                    st.success(f"Disconnected from {template.name}")
                                    st.rerun()
                                except:
                                    pass
                                finally:
                                    loop.close()
        
        st.divider()
        
        # Connection Status
        st.subheader("📊 Active Connections")
        if assistant.active_servers:
            for server_name, adapter in assistant.active_servers.items():
                st.markdown(f"✅ {adapter.server_info.icon} **{adapter.server_info.name}**")
        else:
            st.info("No servers connected")
        
        st.divider()
        
        # LangChain Tools Status
        if api_key and hasattr(assistant, 'langchain_tools') and assistant.langchain_tools:
            st.subheader("🛠️ LangChain Tools")
            for tool in assistant.langchain_tools:
                st.write(f"• {tool.name}")
        
        st.divider()
        
        # Quick Actions
        st.subheader("💡 Quick Actions")
        
        if assistant.active_servers:
            quick_actions = {}
            
            for server_name, adapter in assistant.active_servers.items():
                server_info = adapter.server_info
                if server_name == "github":
                    quick_actions["🐙 GitHub Commits"] = "Show my latest GitHub commits"
                elif server_name == "gmail":
                    quick_actions["📧 Check Gmail"] = "Check my recent emails"
            
            if len(assistant.active_servers) >= 2:
                quick_actions["🔄 Multi-Service"] = "Run a workflow across multiple services"
                quick_actions["🧠 Smart Analysis"] = "Use LangChain to analyze and plan my workflow"
            
            for action_name, action_prompt in quick_actions.items():
                if st.button(action_name, key=f"quick_{action_name}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": action_prompt})
                    st.rerun()
        else:
            st.info("Connect servers to see quick actions")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    if "🧠 LangChain:" in message["content"]:
                        st.markdown(f'<div class="langchain-response">{message["content"]}</div>', unsafe_allow_html=True)
                    elif "🔄" in message["content"]:
                        st.markdown(f'<div class="tool-response">{message["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])
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
                    
                    async def process_message():
                        response_parts = []
                        async for chunk in assistant.process_request(prompt):
                            response_parts.append(chunk)
                            current_response = "\n".join(response_parts)
                            
                            if "🧠 LangChain:" in chunk:
                                message_placeholder.markdown(f'<div class="langchain-response">{current_response}</div>', unsafe_allow_html=True)
                            elif "🔄" in chunk:
                                message_placeholder.markdown(f'<div class="workflow-step">{current_response}</div>', unsafe_allow_html=True)
                            else:
                                message_placeholder.markdown(current_response)
                        
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
                    
                    # Show server URL (masked for privacy)
                    masked_url = adapter.server_info.url[:30] + "..." if len(adapter.server_info.url) > 30 else adapter.server_info.url
                    st.write(f"**URL**: `{masked_url}`")
        else:
            st.info("No tools available. Connect to MCP servers first.")
        
        st.divider()
        
        # LangChain Status
        st.subheader("🔗 LangChain Status")
        if assistant.api_key:
            st.success("✅ API Key: Connected")
            st.info(f"🤖 Model: {assistant.model}")
            
            if assistant.agent:
                st.success(f"🛠️ Agent: Active ({len(assistant.langchain_tools)} tools)")
            else:
                st.warning("🛠️ Agent: Inactive (no tools)")
                
            if assistant.memory:
                st.info(f"🧠 Memory: Active")
        else:
            st.error("❌ API Key: Not provided")
        
        st.divider()
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        **With LangChain Enhancement:**
        1. **Add API Key**: Enter your OpenRouter API key for LLM processing
        2. **Get MCP URLs**: Create servers at Composio and copy the HTTPS stream URLs
        3. **Connect Servers**: Add and connect to your MCP servers
        4. **Smart Chat**: Use natural language - LangChain will analyze and plan
        5. **Advanced Workflows**: Let the AI agent coordinate multiple tools
        
        **LangChain Features:**
        - 🧠 Intelligent request analysis
        - 🛠️ Automatic tool selection
        - 🔄 Multi-step workflow planning
        - 💭 Conversational memory
        - 🎯 Context-aware responses
        """)
        
        st.divider()
        
        # Example Prompts
        st.subheader("💡 Example Prompts")
        example_prompts = [
            "📊 Create a summary of my GitHub activity and send it via email",
            "📧 Check my recent emails and create GitHub issues for any bug reports",
            "🔄 Set up an automated workflow for daily GitHub activity reports",
            "🧠 Analyze my GitHub repositories and email me a project status update"
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
        st.metric("LangChain Tools", len(getattr(assistant, 'langchain_tools', [])))
    
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if assistant.memory:
                assistant.memory.clear()
            st.rerun()

if __name__ == "__main__":
    main()
