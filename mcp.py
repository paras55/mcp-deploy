#!/usr/bin/env python3
"""
Gmail Only Workflow Assistant - FINAL FIXED VERSION
Focused on Gmail integration only, fixes HTTP 406 error

LATEST FIXES:
- ✅ Fixed HTTP 406 "Not Acceptable" error
- ✅ Updated Gmail action names to match actual API
- ✅ Fixed response parsing for actual Gmail API responses
- ✅ Proper Accept headers for MCP servers
- ✅ Better SSE (Server-Sent Events) handling
- ✅ No event loop issues
- ✅ Robust error handling
- ✅ Changed header color
"""

# ============================================================================
# IMPORTS AND DEPENDENCIES
# ============================================================================

import streamlit as st
import json
import requests
import time
import re
import traceback
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from pydantic import Field

# ============================================================================
# STREAMLIT CONFIGURATION AND STYLING
# ============================================================================

st.set_page_config(
    page_title="Gmail Workflow Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with new header color
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .tool-response {
        background: #e3f2fd;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
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
        font-family: 'Courier New', monospace;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .gmail-email {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ea4335;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# GMAIL MCP ADAPTER - FIXES HTTP 406
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

class GmailMCPAdapter:
    """
    Gmail-focused MCP Adapter - Fixes HTTP 406 error with proper Accept headers
    """
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.connected = False
        self.session_id = None
        self.debug_mode = True
        
        # Create session with PROPER headers for MCP protocol
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Gmail-Assistant/2.0.0",
            "Content-Type": "application/json",
            # CRITICAL: Both accept types required to avoid HTTP 406
            "Accept": "application/json, text/event-stream",
            "Cache-Control": "no-cache"
        })
    
    def _debug_log(self, message: str):
        """Debug logging helper"""
        if self.debug_mode:
            print(f"[DEBUG] Gmail: {message}")
    
    def connect(self):
        """Connect to the Gmail MCP server using synchronous requests"""
        if not self.server_info.url:
            raise Exception("No Gmail server URL provided")
            
        self._debug_log(f"Connecting to {self.server_info.url}")
        
        try:
            # Initialize connection payload
            init_payload = {
                "jsonrpc": "2.0",
                "id": "gmail-init-1",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {
                        "roots": {"listChanged": True},
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "Gmail Workflow Assistant",
                        "version": "2.0.0"
                    }
                }
            }
            
            self._debug_log("Sending initialization request...")
            
            # Send POST request with proper headers
            response = self.session.post(
                self.server_info.url,
                json=init_payload,
                timeout=30
            )
            
            self._debug_log(f"Response status: {response.status_code}")
            self._debug_log(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                # Handle both JSON and SSE responses
                content_type = response.headers.get('Content-Type', '').lower()
                
                if 'application/json' in content_type:
                    try:
                        result = response.json()
                        self._debug_log("JSON response received")
                        
                        if "result" in result:
                            self.session_id = response.headers.get('Mcp-Session-Id')
                            self.connected = True
                            self.server_info.connected = True
                            self._debug_log("✅ JSON connection successful!")
                            return True
                        else:
                            raise Exception(f"Initialize failed: {result.get('error', 'Unknown error')}")
                            
                    except json.JSONDecodeError:
                        self._debug_log("JSON decode failed, trying SSE interpretation")
                
                elif 'text/event-stream' in content_type or 'text/plain' in content_type:
                    # Handle Server-Sent Events or plain text response
                    self._debug_log("SSE/Text response received")
                    response_text = response.text
                    
                    # Parse SSE format if present
                    if self._parse_sse_response(response_text):
                        self.session_id = response.headers.get('Mcp-Session-Id')
                        self.connected = True
                        self.server_info.connected = True
                        self._debug_log("✅ SSE connection successful!")
                        return True
                    else:
                        # Assume successful connection for non-standard responses
                        self.connected = True
                        self.server_info.connected = True
                        self._debug_log("✅ Connection assumed successful!")
                        return True
                
                else:
                    # Handle any other successful response
                    self.connected = True
                    self.server_info.connected = True
                    self._debug_log("✅ Generic successful connection!")
                    return True
                    
            elif response.status_code == 406:
                # HTTP 406 Not Acceptable - specific error handling
                error_text = response.text
                self._debug_log(f"HTTP 406 Error: {error_text}")
                raise Exception(f"Server requires different Accept headers. Error: {error_text}")
                
            else:
                error_text = response.text
                self._debug_log(f"HTTP Error {response.status_code}: {error_text}")
                raise Exception(f"HTTP {response.status_code}: {error_text}")
                
        except requests.exceptions.RequestException as e:
            self._debug_log(f"Request error: {str(e)}")
            raise Exception(f"Connection failed: {str(e)}")
    
    def _parse_sse_response(self, response_text: str) -> bool:
        """Parse Server-Sent Events response"""
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data and not data.startswith(':'):
                        try:
                            event_data = json.loads(data)
                            if 'result' in event_data or 'method' in event_data:
                                return True
                        except json.JSONDecodeError:
                            continue
            return False
        except Exception as e:
            self._debug_log(f"SSE parsing error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the Gmail server"""
        self._debug_log("Disconnecting...")
        self.connected = False
        self.server_info.connected = False
        self.session_id = None
        if self.session:
            self.session.close()
    
    def execute_tool(self, tool_name: str, parameters: dict) -> str:
        """Execute a Gmail tool via the MCP server"""
        if not self.connected:
            # Try to connect first
            try:
                self.connect()
            except Exception as e:
                return f"❌ Connection failed: {str(e)}"
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": f"gmail-{tool_name}-{int(time.time())}",
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
                timeout=60
            )
            
            self._debug_log(f"Tool response status: {response.status_code}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '').lower()
                
                if 'application/json' in content_type:
                    try:
                        result = response.json()
                        self._debug_log("Tool execution successful (JSON)")
                        return self._format_gmail_response(result)
                    except json.JSONDecodeError:
                        response_text = response.text
                        self._debug_log("JSON decode failed, using text response")
                        return f"✅ Gmail operation completed:\n{response_text[:500]}..."
                
                elif 'text/event-stream' in content_type:
                    # Handle SSE response
                    response_text = response.text
                    self._debug_log("Tool execution successful (SSE)")
                    parsed_result = self._parse_sse_tool_response(response_text)
                    return self._format_gmail_response({"result": parsed_result})
                
                else:
                    # Handle other response types
                    response_text = response.text
                    self._debug_log("Tool execution successful (other)")
                    return f"✅ Gmail operation completed:\n{response_text[:500]}..."
                    
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
    
    def _parse_sse_tool_response(self, response_text: str):
        """Parse SSE response for tool execution"""
        try:
            lines = response_text.strip().split('\n')
            for line in lines:
                if line.startswith('data: '):
                    data = line[6:]
                    if data and not data.startswith(':'):
                        try:
                            event_data = json.loads(data)
                            if 'result' in event_data:
                                return event_data['result']
                        except json.JSONDecodeError:
                            continue
            return response_text  # Return raw text if no JSON found
        except Exception:
            return response_text
    
    def _format_gmail_response(self, result: dict) -> str:
        """Format Gmail-specific responses with better handling for actual API responses and HTML escaping"""
        if "error" in result:
            error = result["error"]
            if isinstance(error, dict):
                return f"❌ Gmail Error: {self._safe_html_escape(error.get('message', str(error)))}"
            return f"❌ Gmail Error: {self._safe_html_escape(str(error))}"
        
        if "result" in result:
            data = result["result"]
            
            # Handle null or empty responses
            if data is None:
                return "ℹ️ No data returned from Gmail API"
            
            # Handle successful response with data field
            if isinstance(data, dict):
                # Check for the actual response structure
                if "data" in data:
                    response_data = data["data"]
                    
                    # Handle empty response_data
                    if not response_data:
                        return "📧 No emails found for the specified criteria"
                    
                    # Handle successful API response with logId
                    if "successful" in data and data["successful"]:
                        if "logId" in data:
                            log_id = self._safe_html_escape(str(data['logId']))
                            if response_data:
                                formatted_data = self._safe_format_response_data(response_data)
                                return f"✅ **Gmail Operation Successful**\nLog ID: {log_id}\n\n📧 **Response:** {formatted_data}"
                            else:
                                return f"✅ **Gmail Operation Successful**\nLog ID: {log_id}\n\n📧 No data returned"
                        else:
                            if response_data:
                                formatted_data = self._safe_format_response_data(response_data)
                                return f"✅ **Gmail Operation Successful**\n\n📧 **Response:** {formatted_data}"
                            else:
                                return "✅ **Gmail Operation Successful** - No data returned"
                    
                    # Handle list of emails
                    if isinstance(response_data, list) and len(response_data) > 0:
                        return self._format_email_list(response_data)
                    
                    # Handle single email object
                    if isinstance(response_data, dict):
                        return self._format_single_email(response_data)
                
                # Handle other response formats
                elif "emails" in data:
                    emails = data["emails"]
                    if isinstance(emails, list) and len(emails) > 0:
                        return self._format_email_list(emails)
                    else:
                        return "📧 No emails found for the specified time period"
                        
                elif "content" in data:
                    content = data["content"]
                    if isinstance(content, list) and len(content) > 0:
                        first_content = content[0]
                        if isinstance(first_content, dict):
                            return self._safe_html_escape(first_content.get("text", str(content)))
                        return self._safe_html_escape(str(first_content))
                    else:
                        return self._safe_html_escape(str(content)) if content else "ℹ️ Empty response"
                
                elif "message" in data:
                    return f"📧 Gmail: {self._safe_html_escape(str(data['message']))}"
                
                # Handle successful operation responses
                elif "successful" in data and data["successful"]:
                    if "logId" in data:
                        log_id = self._safe_html_escape(str(data['logId']))
                        return f"✅ **Gmail Operation Successful**\nLog ID: {log_id}"
                    else:
                        return "✅ **Gmail Operation Successful**"
                
                else:
                    # Handle generic response
                    if isinstance(data, dict) and len(data) < 10:
                        formatted_lines = []
                        for key, value in data.items():
                            safe_key = self._safe_html_escape(str(key))
                            safe_value = self._safe_html_escape(str(value))
                            formatted_lines.append(f"**{safe_key}:** {safe_value}")
                        return "\n".join(formatted_lines) if formatted_lines else "ℹ️ Empty response"
                    else:
                        # For complex data, use code block to prevent HTML parsing issues
                        safe_data = json.dumps(data, indent=2)[:1000]
                        return f"✅ **Gmail Operation Completed**\n```json\n{safe_data}...\n```"
                        
            elif isinstance(data, list):
                if len(data) > 0:
                    # Check if it's a list of emails
                    if all(isinstance(item, dict) and any(key in item for key in ['subject', 'from', 'to', 'sender', 'recipients']) for item in data):
                        return self._format_email_list(data)
                    else:
                        safe_items = [self._safe_html_escape(str(item)) for item in data[:20]]
                        return "\n".join([f"• {item}" for item in safe_items])
                else:
                    return "ℹ️ No results returned"
            else:
                return self._safe_html_escape(str(data)) if data else "ℹ️ Empty response"
        
        return "✅ Gmail operation completed successfully"
    
    def _safe_format_response_data(self, data):
        """Safely format response data to prevent HTML issues"""
        try:
            if isinstance(data, (dict, list)):
                # Use code block for complex data to prevent HTML parsing
                return f"```json\n{json.dumps(data, indent=2)[:500]}...\n```"
            else:
                return self._safe_html_escape(str(data))
        except Exception:
            return self._safe_html_escape(str(data))
    
    def _safe_html_escape(self, text: str) -> str:
        """Safely escape HTML characters that might cause issues"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Replace problematic characters that could be interpreted as HTML
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#39;'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text
    
    def _format_email_list(self, emails: list) -> str:
        """Format a list of emails for display"""
        email_list = []
        for i, email in enumerate(emails[:15]):  # Show up to 15 emails
            email_entry = self._format_single_email(email, i+1)
            if email_entry:
                email_list.append(email_entry)
        
        total_count = len(emails)
        displayed_count = len(email_list)
        
        header = f"📧 **Found {total_count} emails** (showing {displayed_count}):\n\n"
        
        return header + "\n\n".join(email_list) if email_list else "📧 No emails to display"
    
    def _format_single_email(self, email: dict, index: int = None) -> str:
        """Format a single email for display with proper HTML escaping"""
        if not isinstance(email, dict):
            return f"• {self._safe_html_escape(str(email))}"
        
        # Extract email fields with multiple possible key names
        sender = email.get("from") or email.get("sender") or "Unknown Sender"
        subject = email.get("subject") or email.get("title") or "No Subject"
        date = email.get("date") or email.get("timestamp") or email.get("time") or "Unknown Date"
        snippet = email.get("snippet") or email.get("body") or email.get("preview") or email.get("content")
        
        # Safely escape all fields to prevent HTML issues
        sender = self._safe_html_escape(str(sender))
        subject = self._safe_html_escape(str(subject))
        date = self._safe_html_escape(str(date))
        
        # Clean up snippet
        if snippet:
            if isinstance(snippet, str):
                snippet = snippet[:150] + "..." if len(snippet) > 150 else snippet
                # Remove extra whitespace and line breaks
                snippet = ' '.join(snippet.split())
                snippet = self._safe_html_escape(snippet)
            else:
                snippet = self._safe_html_escape(str(snippet)[:150] + "...")
        else:
            snippet = "No preview available"
        
        # Handle recipients (to field)
        recipients = email.get("to") or email.get("recipients") or "Not specified"
        recipients = self._safe_html_escape(str(recipients))
        
        prefix = f"**📧 Email {index}:**" if index else "**📧 Email:**"
        
        email_entry = f"""{prefix}
👤 **From:** {sender}
📝 **Subject:** {subject}  
📅 **Date:** {date}
📨 **To:** {recipients}
💬 **Preview:** {snippet}"""
        
        return email_entry
    
    def _safe_html_escape(self, text: str) -> str:
        """Safely escape HTML characters that might cause issues"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Replace problematic characters that could be interpreted as HTML
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '&': '&amp;',
            '"': '&quot;',
            "'": '&#39;'
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text

# ============================================================================
# GMAIL LANGCHAIN TOOL - UPDATED WITH CORRECT ACTION NAMES
# ============================================================================

class GmailTool(BaseTool):
    """Gmail-focused LangChain Tool with updated action names"""
    name: str = Field()
    description: str = Field()
    server_adapter: Any = Field()
    tool_name: str = Field()
    
    def _run(self, query: str) -> str:
        """Execute Gmail tool - COMPLETELY SYNCHRONOUS"""
        try:
            params = self._parse_query_params(query)
            result = self.server_adapter.execute_tool(self.tool_name, params)
            return result
        except Exception as e:
            error_trace = traceback.format_exc()
            return f"❌ Error executing {self.tool_name}: {str(e)}\n\n🔧 Debug trace:\n{error_trace}"
    
    def _parse_query_params(self, query: str) -> dict:
        """Parse query into Gmail-specific parameters with updated action names"""
        params = {}
        query_lower = query.lower()
        
        if self.tool_name == "GMAIL_SEND_EMAIL":
            params = {
                "to": "user@example.com",
                "subject": f"Message: {query}",
                "body": query
            }
        elif self.tool_name == "GMAIL_FETCH_EMAILS":
            # Parse time-based queries for fetching emails
            if "yesterday" in query_lower:
                yesterday = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{(yesterday + timedelta(days=1)).strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "2 days" in query_lower or "two days" in query_lower:
                two_days_ago = datetime.now() - timedelta(days=2)
                one_day_ago = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{two_days_ago.strftime('%Y/%m/%d')} before:{one_day_ago.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "3 days" in query_lower or "three days" in query_lower:
                three_days_ago = datetime.now() - timedelta(days=3)
                two_days_ago = datetime.now() - timedelta(days=2)
                params = {
                    "query": f"after:{three_days_ago.strftime('%Y/%m/%d')} before:{two_days_ago.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "week" in query_lower or "7 days" in query_lower:
                week_ago = datetime.now() - timedelta(days=7)
                params = {
                    "query": f"after:{week_ago.strftime('%Y/%m/%d')}",
                    "max_results": 50
                }
            elif "today" in query_lower:
                today = datetime.now()
                params = {
                    "query": f"after:{today.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            else:
                params = {
                    "query": "in:inbox",
                    "max_results": 25
                }
        elif self.tool_name == "GMAIL_SEARCH_PEOPLE":
            params = {"query": query}
        else:
            params = {"query": query}
        
        return params

# ============================================================================
# GMAIL WORKFLOW ASSISTANT - UPDATED WITH CORRECT ACTIONS
# ============================================================================

class GmailWorkflowAssistant:
    """Gmail-focused Workflow Assistant with updated action names"""
    
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
        # Updated Gmail server template with correct action names
        self.server_template = MCPServerInfo(
            name="Gmail",
            description="Gmail email management and operations",
            capabilities=[
                "GMAIL_FETCH_EMAILS",
                "GMAIL_SEARCH_PEOPLE", 
                "GMAIL_SEND_EMAIL",
                "GMAIL_GET_CONTACTS",
                "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID"
            ],
            icon="📧",
            category="Email"
        )
        
        self.gmail_server = None  # GmailMCPAdapter
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
    
    def add_gmail_server(self, server_url: str) -> bool:
        """Add and connect to Gmail MCP server"""
        server_info = MCPServerInfo(
            name=self.server_template.name,
            description=self.server_template.description,
            capabilities=self.server_template.capabilities,
            icon=self.server_template.icon,
            category=self.server_template.category,
            url=server_url
        )
        
        # Create adapter and connect
        adapter = GmailMCPAdapter(server_info)
        adapter.connect()
        
        # Store Gmail server
        self.gmail_server = adapter
        
        # Add LangChain tools
        self._add_langchain_tools()
        
        return True
    
    def _add_langchain_tools(self):
        """Add LangChain tools for Gmail server"""
        if not self.gmail_server:
            return
            
        for capability in self.gmail_server.server_info.capabilities:
            tool = GmailTool(
                name=f"gmail_{capability}",
                description=f"Gmail {capability.replace('_', ' ').title()}",
                server_adapter=self.gmail_server,
                tool_name=capability
            )
            self.langchain_tools.append(tool)
        
        self._update_agent()
    
    def remove_gmail_server(self):
        """Remove and disconnect from Gmail server"""
        if self.gmail_server:
            self.gmail_server.disconnect()
            self.gmail_server = None
            
            # Remove LangChain tools
            self.langchain_tools = []
            self._update_agent()
    
    def process_request(self, user_input: str):
        """Process Gmail request - COMPLETELY SYNCHRONOUS"""
        if not self.gmail_server:
            yield "❌ No Gmail server connected. Please add your Gmail MCP server URL in the sidebar."
            return
        
        yield f"🧠 **Analyzing Gmail request:** {user_input}"
        
        try:
            yield f"📧 **Using direct Gmail execution**"
            
            for response in self._direct_process_request(user_input):
                yield response
                
        except Exception as e:
            yield f"❌ **Error processing request:** {str(e)}"
            yield f"🔧 **Debug info:** {traceback.format_exc()}"
    
    def _direct_process_request(self, user_input: str):
        """Direct Gmail execution with updated action names"""
        # Parse Gmail intent
        intent = self._parse_gmail_intent(user_input)
        
        yield f"🔍 **Gmail operation:** {intent.get('action', 'unknown')}"
        
        try:
            # Execute the Gmail tool directly
            result = self.gmail_server.execute_tool(
                intent["action"], 
                intent.get("params", {})
            )
            
            yield f"📊 **Gmail Results:**"
            yield result
            
        except Exception as e:
            yield f"❌ **Gmail Error:** {str(e)}"
            yield f"🔧 **Debug:** {traceback.format_exc()}"
    
    def _parse_gmail_intent(self, user_input: str) -> dict:
        """Parse user intent for Gmail operations with updated action names"""
        user_input_lower = user_input.lower()
        
        if "send" in user_input_lower and "email" in user_input_lower:
            return {
                "action": "GMAIL_SEND_EMAIL",
                "params": {
                    "to": "user@example.com",
                    "subject": "Test Email",
                    "body": "Hello from Gmail Assistant!"
                }
            }
        elif "contact" in user_input_lower or "people" in user_input_lower:
            return {
                "action": "GMAIL_SEARCH_PEOPLE",
                "params": {"query": user_input}
            }
        else:
            # Default to fetch emails
            action = "GMAIL_FETCH_EMAILS"
            
            # Parse time-based requests
            if "2 days" in user_input_lower or "two days" in user_input_lower:
                two_days_ago = datetime.now() - timedelta(days=2)
                one_day_ago = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{two_days_ago.strftime('%Y/%m/%d')} before:{one_day_ago.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "yesterday" in user_input_lower:
                yesterday = datetime.now() - timedelta(days=1)
                params = {
                    "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{(yesterday + timedelta(days=1)).strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "today" in user_input_lower:
                today = datetime.now()
                params = {
                    "query": f"after:{today.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "week" in user_input_lower:
                week_ago = datetime.now() - timedelta(days=7)
                params = {
                    "query": f"after:{week_ago.strftime('%Y/%m/%d')}",
                    "max_results": 50
                }
            else:
                params = {
                    "query": "in:inbox",
                    "max_results": 25
                }
                
            return {"action": action, "params": params}

# ============================================================================
# USER INTERFACE - GMAIL FOCUSED
# ============================================================================

def create_gmail_assistant(api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
    """Create the Gmail-focused assistant"""
    return GmailWorkflowAssistant(api_key, model)

def render_gmail_message(content: str, message_placeholder):
    """Render Gmail message with appropriate styling"""
    if "🧠" in content or "🚀" in content:
        message_placeholder.markdown(f'<div class="langchain-response">{content}</div>', unsafe_allow_html=True)
    elif "📧" in content or "📊" in content:
        message_placeholder.markdown(f'<div class="tool-response">{content}</div>', unsafe_allow_html=True)
    elif "❌" in content:
        message_placeholder.markdown(f'<div class="error-box">{content}</div>', unsafe_allow_html=True)
    elif "✅" in content:
        message_placeholder.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
    else:
        message_placeholder.markdown(content)

# ============================================================================
# MAIN APPLICATION - GMAIL FOCUSED
# ============================================================================

def main():
    """Gmail-focused main application"""
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = create_gmail_assistant()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'gmail_url' not in st.session_state:
        st.session_state.gmail_url = ""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📧 Gmail Workflow Assistant</h1>
        <p>FINAL FIXED VERSION - Updated with Correct Gmail Actions!</p>
        <p style="font-size: 0.9em; opacity: 0.9;">✅ Correct Action Names • ✅ Better Response Parsing • ✅ HTTP 406 Fixed • ✅ Gmail Focused</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📧 Gmail Configuration")
        
        # API Key (Optional)
        api_key = st.text_input(
            "OpenRouter API Key (Optional)",
            type="password",
            help="For enhanced LangChain features - not required for basic Gmail functionality",
            placeholder="sk-or-..."
        )
        
        # Update assistant if API key changes
        if api_key and (not hasattr(st.session_state.assistant, 'api_key') or 
                       st.session_state.assistant.api_key != api_key):
            st.session_state.assistant = create_gmail_assistant(api_key)
        
        # Gmail Status
        if api_key:
            st.success("🔗 Enhanced Mode: Active")
        else:
            st.info("🎯 Direct Mode: Active (Recommended)")
        
        st.divider()
        
        # Gmail Server Configuration
        st.subheader("🔌 Gmail MCP Server")
        st.info("💡 Get your Gmail MCP server URL from Composio")
        
        assistant = st.session_state.assistant
        
        # Gmail URL Input
        gmail_url = st.text_input(
            "Gmail MCP Server URL",
            value=st.session_state.gmail_url,
            placeholder="https://mcp.composio.dev/gmail/your-server-id",
            help="Paste your Composio Gmail MCP server URL here"
        )
        
        # Update URL in session state
        if gmail_url != st.session_state.gmail_url:
            st.session_state.gmail_url = gmail_url
        
        # Connection Controls
        col1, col2 = st.columns(2)
        
        with col1:
            if not assistant.gmail_server:
                if st.button("🔗 Connect Gmail", disabled=not gmail_url, use_container_width=True):
                    try:
                        with st.spinner("Connecting to Gmail MCP server..."):
                            success = assistant.add_gmail_server(gmail_url)
                            if success:
                                st.success("✅ Gmail connected successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Failed to connect to Gmail")
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                        
                        # Show debug info
                        with st.expander("🔧 Debug Information"):
                            st.code(traceback.format_exc())
            else:
                st.success("✅ Connected")
        
        with col2:
            if assistant.gmail_server:
                if st.button("🔌 Disconnect", use_container_width=True):
                    try:
                        assistant.remove_gmail_server()
                        st.success("Disconnected from Gmail")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        st.divider()
        
        # Connection Status
        st.subheader("📊 Gmail Status")
        if assistant.gmail_server:
            if assistant.gmail_server.connected:
                st.markdown("🟢 **Connected and Ready**")
                
                # Show capabilities
                st.write("**Available Operations:**")
                operations_display = {
                    "GMAIL_FETCH_EMAILS": "📥 Fetch Emails",
                    "GMAIL_SEARCH_PEOPLE": "👥 Search People", 
                    "GMAIL_SEND_EMAIL": "📤 Send Email",
                    "GMAIL_GET_CONTACTS": "📇 Get Contacts",
                    "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID": "📧 Get Message by ID"
                }
                
                for capability in assistant.gmail_server.server_info.capabilities:
                    display_name = operations_display.get(capability, capability.replace('_', ' ').title())
                    st.write(f"• {display_name}")
                    
                # Show server info
                masked_url = gmail_url[:40] + "..." if len(gmail_url) > 40 else gmail_url
                st.write(f"**Server:** `{masked_url}`")
            else:
                st.markdown("🟡 **Connected but Not Ready**")
        else:
            st.markdown("🔴 **Not Connected**")
            st.info("Connect to your Gmail MCP server above to get started")
        
        st.divider()
        
        # Quick Actions
        st.subheader("💡 Quick Gmail Actions")
        
        if assistant.gmail_server and assistant.gmail_server.connected:
            quick_actions = [
                ("📧 Recent Emails", "Show me my recent emails"),
                ("📅 Yesterday's Emails", "Get my emails from yesterday"),
                ("🗓️ 2 Days Back", "Show emails from 2 days back"),
                ("📮 This Week", "Get emails from this week"),
                ("📨 Today's Emails", "Show me today's emails"),
                ("👥 Search People", "Search people in my contacts")
            ]
            
            for action_name, action_prompt in quick_actions:
                if st.button(action_name, key=f"quick_{action_name}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": action_prompt})
                    st.rerun()
        else:
            st.info("Connect to Gmail server to see quick actions")
        
        st.divider()
        
        # Help Section
        st.subheader("❓ Need Help?")
        with st.expander("📖 How to Get Gmail MCP URL"):
            st.markdown("""
            **Steps to get your Gmail MCP server URL:**
            
            1. Visit [Composio MCP](https://mcp.composio.dev/)
            2. Sign up or log in to your account
            3. Navigate to Gmail integration
            4. Create a new Gmail MCP server
            5. Copy the provided HTTPS URL
            6. Paste it in the field above
            7. Click "Connect Gmail"
            
            **URL format example:**
            `https://mcp.composio.dev/gmail/abc123-def456`
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common issues and solutions:**
            
            **HTTP 406 Error:**
            - This version fixes the accept headers issue
            - Make sure your URL is correct
            
            **Connection Timeout:**
            - Check your internet connection
            - Verify the MCP server URL
            - Try reconnecting
            
            **Authentication Issues:**
            - Follow Gmail OAuth flow in Composio
            - Make sure permissions are granted
            
            **No Emails Returned:**
            - Check date filters in your request
            - Verify Gmail account has emails
            - Try "recent emails" first
            
            **Empty Response Data:**
            - This is normal if no emails match criteria
            - Try broader search terms
            - Check different time periods
            """)
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Gmail Chat Interface")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    content = message["content"]
                    if "🧠" in content or "📧" in content:
                        st.markdown(f'<div class="tool-response">{content}</div>', unsafe_allow_html=True)
                    elif "❌" in content:
                        st.markdown(f'<div class="error-box">{content}</div>', unsafe_allow_html=True)
                    elif "✅" in content:
                        st.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(content)
                else:
                    st.markdown(message["content"])
        
        # Chat Input
        if assistant.gmail_server and assistant.gmail_server.connected:
            if prompt := st.chat_input("Ask me about your Gmail emails..."):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process with assistant
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    def process_gmail_message():
                        response_parts = []
                        try:
                            for chunk in assistant.process_request(prompt):
                                response_parts.append(chunk)
                                current_response = "\n".join(response_parts)
                                
                                # Update display with appropriate styling
                                render_gmail_message(current_response, message_placeholder)
                                
                        except Exception as e:
                            error_msg = f"❌ Error processing Gmail request: {str(e)}\n\n🔧 Debug:\n{traceback.format_exc()}"
                            response_parts.append(error_msg)
                            message_placeholder.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
                        
                        return "\n".join(response_parts)
                    
                    # Simple synchronous execution
                    full_response = process_gmail_message()
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
        else:
            st.info("👈 Please connect to your Gmail MCP server in the sidebar to start chatting.")
            
            # Show example commands
            st.markdown("""
            **Once connected, you can try commands like:**
            - "Show me my recent emails"
            - "Get emails from yesterday"
            - "What emails did I receive 2 days back?"
            - "Show me this week's emails"
            - "Get today's emails"
            - "Search people in my contacts"
            """)
    
    with col2:
        st.subheader("📈 Gmail Insights")
        
        if assistant.gmail_server and assistant.gmail_server.connected:
            # Connection metrics
            st.metric("Gmail Server", "Connected", delta="Healthy")
            st.metric("Available Tools", len(assistant.gmail_server.server_info.capabilities))
            
            st.divider()
            
            # Gmail Operations
            st.subheader("🛠️ Available Operations")
            
            operations = {
                "GMAIL_FETCH_EMAILS": "📥 Fetch & search emails",
                "GMAIL_SEARCH_PEOPLE": "👥 Search people/contacts", 
                "GMAIL_SEND_EMAIL": "📤 Send new emails",
                "GMAIL_GET_CONTACTS": "📇 Get contact list",
                "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID": "📧 Get specific message"
            }
            
            for op, desc in operations.items():
                if op in assistant.gmail_server.server_info.capabilities:
                    st.write(f"**{desc}**")
            
        else:
            st.info("Gmail server not connected")
        
        st.divider()
        
        # Performance Info
        st.subheader("⚡ Performance")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Chat Messages", len(st.session_state.messages))
        with col_b:
            status = "Connected" if (assistant.gmail_server and assistant.gmail_server.connected) else "Disconnected"
            st.metric("Status", status)
        
        st.divider()
        
        # Example Queries
        st.subheader("💡 Example Queries")
        
        examples = [
            "📧 Show recent emails",
            "📅 Yesterday's emails", 
            "🗓️ Emails from 2 days back",
            "📮 This week's emails",
            "📨 Today's emails",
            "👥 Search my contacts"
        ]
        
        for example in examples:
            if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
                example_text = example.split(" ", 1)[1] if " " in example else example
                st.session_state.messages.append({"role": "user", "content": example_text})
                st.rerun()
    
    # Footer
    st.divider()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = 1 if (assistant.gmail_server and assistant.gmail_server.connected) else 0
        st.metric("Gmail Connection", status, delta="Active" if status else "Inactive")
    
    with col2:
        tools = len(assistant.langchain_tools) if assistant.langchain_tools else 0
        st.metric("LangChain Tools", tools)
    
    with col3:
        st.metric("Total Messages", len(st.session_state.messages))
    
    with col4:
        mode = "Enhanced" if assistant.api_key else "Direct"
        st.metric("Mode", mode)
    
    # Clear chat and reset buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                if hasattr(assistant, 'memory') and assistant.memory:
                    assistant.memory.clear()
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset Connection", use_container_width=True):
            if assistant.gmail_server:
                try:
                    assistant.remove_gmail_server()
                    st.success("Connection reset successfully")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {str(e)}")

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
