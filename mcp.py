#!/usr/bin/env python3
"""
Enhanced Gmail Workflow Assistant - FULL FIXED VERSION
- Fixes missing email display (supports list and dict responses)
- Always returns email cards with proper HTML for Streamlit rendering
- Debug logging for raw Gmail API response
- Retains all existing features (CSS, LangChain, enhanced UI)
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
import html
from bs4 import BeautifulSoup

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
    page_title="Enhanced Gmail Assistant",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLES ---
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
    .email-card {
        background: #ffffff;
        border: 1px solid #e1e5e9;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .email-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    .email-header {
        border-bottom: 1px solid #f0f2f5;
        padding-bottom: 0.8rem;
        margin-bottom: 0.8rem;
    }
    .email-from {
        color: #1a73e8;
        font-weight: 600;
        font-size: 1.1em;
    }
    .email-subject {
        color: #202124;
        font-weight: 700;
        font-size: 1.2em;
        margin: 0.4rem 0;
    }
    .email-date {
        color: #5f6368;
        font-size: 0.9em;
    }
    .email-preview {
        background: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #1a73e8;
        font-style: italic;
        color: #495057;
        margin-top: 0.8rem;
    }
    .email-labels {
        margin-top: 0.6rem;
    }
    .label-badge {
        display: inline-block;
        background: #e8f0fe;
        color: #1967d2;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8em;
        margin-right: 0.4rem;
        margin-bottom: 0.2rem;
    }
    .tool-response {
        background: #e3f2fd;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #2196f3;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 3px solid #dc3545;
        margin: 0.5rem 0;
    }
    .stats-container {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .no-emails {
        text-align: center;
        color: #6c757d;
        font-style: italic;
        padding: 2rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# EMAIL CONTENT PROCESSING UTILITIES
# ============================================================================
class EmailContentProcessor:
    @staticmethod
    def clean_html_content(html_content: str) -> str:
        if not html_content:
            return ""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception:
            clean_text = re.sub('<[^<]+?>', '', html_content)
            clean_text = html.unescape(clean_text)
            return ' '.join(clean_text.split())

    @staticmethod
    def extract_email_preview(content: str, max_length: int = 150) -> str:
        if not content:
            return "No preview available"
        if '<' in content and '>' in content:
            content = EmailContentProcessor.clean_html_content(content)
        content = ' '.join(content.split())
        if len(content) > max_length:
            content = content[:max_length].rsplit(' ', 1)[0] + "..."
        return content or "No preview available"

    @staticmethod
    def format_email_date(date_str: str) -> str:
        if not date_str:
            return "Unknown date"
        try:
            date_formats = [
                "%a, %d %b %Y %H:%M:%S %z",
                "%d %b %Y %H:%M:%S %z", 
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ"
            ]
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt)
                    break
                except ValueError:
                    continue
            if parsed_date:
                now = datetime.now()
                if parsed_date.tzinfo:
                    parsed_date = parsed_date.replace(tzinfo=None)
                diff = now - parsed_date
                if diff.days == 0:
                    return f"Today at {parsed_date.strftime('%I:%M %p')}"
                elif diff.days == 1:
                    return f"Yesterday at {parsed_date.strftime('%I:%M %p')}"
                elif diff.days < 7:
                    return f"{diff.days} days ago"
                else:
                    return parsed_date.strftime("%b %d, %Y")
        except Exception:
            pass
        return str(date_str)[:50]

    @staticmethod
    def clean_email_address(address: str) -> str:
        if not address:
            return "Unknown"
        address = str(address).strip()
        if '<' in address and '>' in address:
            match = re.search(r'<([^>]+)>', address)
            if match:
                email = match.group(1).strip()
                name_part = address.split('<')[0].strip().strip('"\'')
                if name_part and name_part != email:
                    return f"{name_part} <{email}>"
                return email
        return address

# ============================================================================
# ENHANCED GMAIL MCP ADAPTER (FIXED)
# ============================================================================
@dataclass
class MCPServerInfo:
    name: str
    description: str
    capabilities: List[str]
    icon: str
    category: str
    url: str = ""
    connected: bool = False

class EnhancedGmailMCPAdapter:
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.connected = False
        self.session_id = None
        self.debug_mode = True
        self.content_processor = EmailContentProcessor()
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Enhanced-Gmail-Assistant/2.1.1",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Cache-Control": "no-cache"
        })

    def _debug_log(self, message: str):
        if self.debug_mode:
            print(f"[DEBUG] Enhanced Gmail: {message}")

    def connect(self):
        if not self.server_info.url:
            raise Exception("No Gmail server URL provided")
        init_payload = {
            "jsonrpc": "2.0",
            "id": "gmail-init-1",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "Enhanced Gmail Assistant","version": "2.1.1"}
            }
        }
        response = self.session.post(self.server_info.url, json=init_payload, timeout=30)
        if response.status_code == 200:
            self.connected = True
            self.server_info.connected = True
            return True
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    def execute_tool(self, tool_name: str, parameters: dict) -> str:
        if not self.connected:
            self.connect()
        payload = {
            "jsonrpc": "2.0",
            "id": f"gmail-{tool_name}-{int(time.time())}",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": parameters}
        }
        response = self.session.post(self.server_info.url, json=payload, timeout=60)
        if response.status_code == 200:
            try:
                result = response.json()
                self._debug_log(f"Raw Gmail result: {json.dumps(result, indent=2)}")
                return self._format_enhanced_response(result, tool_name)
            except json.JSONDecodeError:
                return self._format_enhanced_response({"result": response.text}, tool_name)
        return f"❌ HTTP {response.status_code}: {response.text}"

    def _format_enhanced_response(self, result: dict, tool_name: str) -> str:
        if "error" in result:
            error = result["error"]
            error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
            return f"❌ **Gmail Error:** {error_msg}"

        if "result" not in result:
            return "ℹ️ No data returned from Gmail API"

        data = result["result"]

        # Handle direct list of emails
        if isinstance(data, list) and data:
            return self._format_email_list(data)

        # Handle dict with emails in various keys
        if isinstance(data, dict):
            emails_data = data.get("data") or data.get("emails") or None
            if isinstance(emails_data, list) and emails_data:
                return self._format_email_list(emails_data)

        return "📧 **No emails found** for the specified criteria"

    def _format_email_list(self, emails: list) -> str:
        if not emails:
            return '<div class="no-emails">📧 No emails to display</div>'
        total_count = len(emails)
        display_count = min(20, total_count)
        header = f"""
        <div class="stats-container">
            <h3>📧 Gmail Results</h3>
            <p><strong>Found:</strong> {total_count} emails | <strong>Showing:</strong> {display_count}</p>
        </div>
        """
        email_cards = []
        for i, email in enumerate(emails[:display_count]):
            email_html = self._format_email_card(email, i + 1)
            if email_html:
                email_cards.append(email_html)
        return header + "\n\n" + "\n".join(email_cards)

    def _format_email_card(self, email: dict, index: int) -> str:
        sender = self.content_processor.clean_email_address(email.get("from") or "Unknown Sender")
        subject = email.get("subject") or "No Subject"
        content = (email.get("messageText") or email.get("snippet") or email.get("body") or "")
        preview = self.content_processor.extract_email_preview(content)
        formatted_date = self.content_processor.format_email_date(email.get("date") or "")
        recipients = email.get("to") or "Not specified"
        if recipients != "Not specified":
            recipients = self.content_processor.clean_email_address(str(recipients))
        labels = email.get("labelIds") or []
        label_badges = ""
        if labels:
            clean_labels = [label for label in labels if label not in ["UNREAD", "INBOX"]]
            if clean_labels:
                label_badges = '<div class="email-labels">' + ''.join(
                    f'<span class="label-badge">{label}</span>' for label in clean_labels[:5]
                ) + '</div>'
        return f"""
        <div class="email-card">
            <div class="email-header">
                <div class="email-from">👤 {html.escape(sender)}</div>
                <div class="email-subject">{html.escape(subject)}</div>
                <div class="email-date">📅 {html.escape(formatted_date)}</div>
                {f'<div style="color: #5f6368; font-size: 0.9em;">📨 To: {html.escape(recipients)}</div>' if recipients != "Not specified" else ''}
            </div>
            <div class="email-preview">
                💬 {html.escape(preview)}
            </div>
            {label_badges}
        </div>
        """
        
        return email_card
    
    def _format_single_email(self, email: dict) -> str:
        """Format single email with enhanced structure"""
        return self._format_email_card(email, 1)

# ============================================================================
# ENHANCED LANGCHAIN TOOL
# ============================================================================

class EnhancedGmailTool(BaseTool):
    """Enhanced Gmail LangChain Tool"""
    name: str = Field()
    description: str = Field()
    server_adapter: Any = Field()
    tool_name: str = Field()
    
    def _run(self, query: str) -> str:
        """Execute Gmail tool with enhanced error handling"""
        try:
            params = self._parse_query_params(query)
            result = self.server_adapter.execute_tool(self.tool_name, params)
            return result
        except Exception as e:
            return f"❌ Error executing {self.tool_name}: {str(e)}"
    
    def _parse_query_params(self, query: str) -> dict:
        """Parse query into Gmail parameters"""
        params = {}
        query_lower = query.lower()
        
        if self.tool_name == "GMAIL_SEND_EMAIL":
            params = {
                "to": "user@example.com",
                "subject": f"Message: {query}",
                "body": query
            }
        elif self.tool_name == "GMAIL_FETCH_EMAILS":
            # Enhanced date parsing
            today = datetime.now()
            
            if "yesterday" in query_lower:
                yesterday = today - timedelta(days=1)
                params = {
                    "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{today.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif any(term in query_lower for term in ["2 days", "two days"]):
                two_days_ago = today - timedelta(days=2)
                yesterday = today - timedelta(days=1)
                params = {
                    "query": f"after:{two_days_ago.strftime('%Y/%m/%d')} before:{yesterday.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif any(term in query_lower for term in ["3 days", "three days"]):
                three_days_ago = today - timedelta(days=3)
                two_days_ago = today - timedelta(days=2)
                params = {
                    "query": f"after:{three_days_ago.strftime('%Y/%m/%d')} before:{two_days_ago.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif any(term in query_lower for term in ["week", "7 days"]):
                week_ago = today - timedelta(days=7)
                params = {
                    "query": f"after:{week_ago.strftime('%Y/%m/%d')}",
                    "max_results": 50
                }
            elif "today" in query_lower:
                params = {
                    "query": f"after:{today.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            else:
                params = {
                    "query": "in:inbox",
                    "max_results": 25
                }
        else:
            params = {"query": query}
        
        return params

# ============================================================================
# ENHANCED GMAIL WORKFLOW ASSISTANT
# ============================================================================

class EnhancedGmailWorkflowAssistant:
    """Enhanced Gmail Workflow Assistant with better formatting"""
    
    def __init__(self, api_key: str = None, model: str = "anthropic/claude-3.5-sonnet"):
        self.server_template = MCPServerInfo(
            name="Gmail",
            description="Enhanced Gmail email management",
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
        
        self.gmail_server = None
        self.api_key = api_key
        self.model = model
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
        """Update the LangChain agent"""
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
        
        adapter = EnhancedGmailMCPAdapter(server_info)
        adapter.connect()
        
        self.gmail_server = adapter
        self._add_langchain_tools()
        
        return True
    
    def _add_langchain_tools(self):
        """Add LangChain tools for Gmail server"""
        if not self.gmail_server:
            return
            
        for capability in self.gmail_server.server_info.capabilities:
            tool = EnhancedGmailTool(
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
            self.langchain_tools = []
            self._update_agent()
    
    def process_request(self, user_input: str):
        """Process Gmail request with enhanced formatting"""
        if not self.gmail_server:
            yield "❌ No Gmail server connected. Please add your Gmail MCP server URL in the sidebar."
            return
        
        yield f"🧠 **Analyzing request:** {user_input}"
        
        try:
            for response in self._enhanced_process_request(user_input):
                yield response
        except Exception as e:
            yield f"❌ **Error:** {str(e)}"
    
    def _enhanced_process_request(self, user_input: str):
        """Enhanced request processing with better output"""
        intent = self._parse_gmail_intent(user_input)
        
        yield f"🔍 **Operation:** {intent.get('action', 'unknown').replace('_', ' ').title()}"
        
        try:
            result = self.gmail_server.execute_tool(
                intent["action"], 
                intent.get("params", {})
            )
            
            yield "📊 **Results:**"
            yield result
            
        except Exception as e:
            yield f"❌ **Gmail Error:** {str(e)}"
    
    def _parse_gmail_intent(self, user_input: str) -> dict:
        """Parse user intent for Gmail operations"""
        user_input_lower = user_input.lower()
        
        if "send" in user_input_lower and "email" in user_input_lower:
            return {
                "action": "GMAIL_SEND_EMAIL",
                "params": {
                    "to": "user@example.com",
                    "subject": "Test Email",
                    "body": "Hello from Enhanced Gmail Assistant!"
                }
            }
        elif "contact" in user_input_lower or "people" in user_input_lower:
            return {
                "action": "GMAIL_SEARCH_PEOPLE",
                "params": {"query": user_input}
            }
        else:
            # Default to fetch emails with enhanced date parsing
            action = "GMAIL_FETCH_EMAILS"
            today = datetime.now()
            
            if any(term in user_input_lower for term in ["2 days", "two days"]):
                two_days_ago = today - timedelta(days=2)
                one_day_ago = today - timedelta(days=1)
                params = {
                    "query": f"after:{two_days_ago.strftime('%Y/%m/%d')} before:{one_day_ago.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "yesterday" in user_input_lower:
                yesterday = today - timedelta(days=1)
                params = {
                    "query": f"after:{yesterday.strftime('%Y/%m/%d')} before:{today.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif "today" in user_input_lower:
                params = {
                    "query": f"after:{today.strftime('%Y/%m/%d')}",
                    "max_results": 25
                }
            elif any(term in user_input_lower for term in ["week", "7 days"]):
                week_ago = today - timedelta(days=7)
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
# MAIN APPLICATION WITH ENHANCED UI
# ============================================================================

def main():
    """Enhanced Gmail Assistant main application"""
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = EnhancedGmailWorkflowAssistant()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'gmail_url' not in st.session_state:
        st.session_state.gmail_url = ""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✨ Enhanced Gmail Assistant</h1>
        <p>Beautiful Email Display with Clean Formatting</p>
        <p style="font-size: 0.9em; opacity: 0.9;">✅ Clean HTML Parsing • ✅ Structured Email Cards • ✅ Enhanced Date Formatting • ✅ Better Error Handling</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📧 Enhanced Gmail Setup")
        
        # API Key (Optional)
        api_key = st.text_input(
            "OpenRouter API Key (Optional)",
            type="password",
            help="For enhanced LangChain features",
            placeholder="sk-or-..."
        )
        
        # Update assistant if API key changes
        if api_key and (not hasattr(st.session_state.assistant, 'api_key') or 
                       st.session_state.assistant.api_key != api_key):
            st.session_state.assistant = EnhancedGmailWorkflowAssistant(api_key)
        
        # Status indicator
        if api_key:
            st.success("🔗 Enhanced Mode: Active")
        else:
            st.info("🎯 Direct Mode: Active (Recommended)")
        
        st.divider()
        
        # Gmail Server Configuration
        st.subheader("🔌 Gmail MCP Server")
        
        assistant = st.session_state.assistant
        
        # Gmail URL Input
        gmail_url = st.text_input(
            "Gmail MCP Server URL",
            value=st.session_state.gmail_url,
            placeholder="https://mcp.composio.dev/gmail/your-server-id",
            help="Get this from Composio Gmail MCP setup"
        )
        
        if gmail_url != st.session_state.gmail_url:
            st.session_state.gmail_url = gmail_url
        
        # Connection Controls
        col1, col2 = st.columns(2)
        
        with col1:
            if not assistant.gmail_server:
                if st.button("🔗 Connect", disabled=not gmail_url, use_container_width=True):
                    try:
                        with st.spinner("Connecting to Gmail..."):
                            success = assistant.add_gmail_server(gmail_url)
                            if success:
                                st.success("✅ Connected!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Connection failed: {str(e)}")
                        with st.expander("🔧 Debug Info"):
                            st.code(traceback.format_exc())
            else:
                st.success("✅ Connected")
        
        with col2:
            if assistant.gmail_server:
                if st.button("🔌 Disconnect", use_container_width=True):
                    assistant.remove_gmail_server()
                    st.success("Disconnected")
                    st.rerun()
        
        st.divider()
        
        # Enhanced Status Display
        st.subheader("📊 Connection Status")
        if assistant.gmail_server and assistant.gmail_server.connected:
            st.markdown("🟢 **Gmail: Connected & Ready**")
            
            # Enhanced capabilities display
            st.write("**📋 Available Operations:**")
            operations = {
                "GMAIL_FETCH_EMAILS": "📥 Fetch & Search Emails",
                "GMAIL_SEARCH_PEOPLE": "👥 Search Contacts", 
                "GMAIL_SEND_EMAIL": "📤 Send New Email",
                "GMAIL_GET_CONTACTS": "📇 Get Contact List",
                "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID": "📧 Get Specific Message"
            }
            
            for capability in assistant.gmail_server.server_info.capabilities:
                display_name = operations.get(capability, capability.replace('_', ' ').title())
                st.write(f"• {display_name}")
        else:
            st.markdown("🔴 **Gmail: Not Connected**")
            st.info("Connect to your Gmail MCP server above")
        
        st.divider()
        
        # Enhanced Quick Actions
        st.subheader("⚡ Quick Actions")
        
        if assistant.gmail_server and assistant.gmail_server.connected:
            quick_actions = [
                ("📬 Recent Emails", "Show me my recent emails"),
                ("📅 Yesterday's Emails", "Get my emails from yesterday"),  
                ("📆 2 Days Back", "Show emails from 2 days back"),
                ("📮 This Week", "Get emails from this week"),
                ("📨 Today's Emails", "Show me today's emails"),
                ("👥 Search People", "Search people in my contacts"),
                ("📤 Send Test Email", "Send a test email")
            ]
            
            for action_name, action_prompt in quick_actions:
                if st.button(action_name, key=f"quick_{hash(action_name)}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": action_prompt})
                    st.rerun()
        else:
            st.info("Connect to Gmail to see quick actions")
        
        st.divider()
        
        # Enhanced Help Section
        st.subheader("❓ Setup Guide")
        
        with st.expander("🚀 Getting Started"):
            st.markdown("""
            **Quick Setup Steps:**
            
            1. **Get Gmail MCP URL:**
               - Visit [Composio MCP](https://mcp.composio.dev/)
               - Create Gmail integration
               - Copy the HTTPS URL
            
            2. **Connect:**
               - Paste URL above
               - Click "Connect"
               - Wait for green status
            
            3. **Start Chatting:**
               - Use quick actions or type requests
               - Get beautifully formatted email results
            """)
        
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **Common Solutions:**
            
            **Raw HTML Output (FIXED):**
            - ✅ This version automatically cleans HTML
            - ✅ Emails now display in clean cards
            
            **Connection Issues:**
            - Verify URL format: `https://mcp.composio.dev/gmail/...`
            - Check internet connection
            - Try reconnecting
            
            **No Emails Found:**
            - Try different time ranges
            - Use "recent emails" first
            - Check Gmail account permissions
            """)
        
        with st.expander("✨ New Features"):
            st.markdown("""
            **Enhanced in this version:**
            
            ✅ **Clean Email Display:**
            - HTML content automatically cleaned
            - Beautiful email cards
            - Readable formatting
            
            ✅ **Better Date Formatting:**
            - "Today at 2:30 PM"
            - "Yesterday at 9:15 AM" 
            - "3 days ago"
            
            ✅ **Enhanced Content:**
            - Smart email previews
            - Clean sender names
            - Label badges
            - Structured layout
            """)
    
    # Main Content Area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Enhanced Gmail Chat")
        
        # Display chat messages with enhanced formatting
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    content = message["content"]
                    
                    # Enhanced message rendering
                    if "email-card" in content:
                        # Render HTML email cards
                        st.markdown(content, unsafe_allow_html=True)
                    elif "🧠" in content or "🔍" in content:
                        st.markdown(f'<div class="tool-response">{content}</div>', unsafe_allow_html=True)
                    elif "❌" in content:
                        st.markdown(f'<div class="error-box">{content}</div>', unsafe_allow_html=True)
                    elif "✅" in content:
                        st.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(content, unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])
        
        # Enhanced Chat Input
        if assistant.gmail_server and assistant.gmail_server.connected:
            if prompt := st.chat_input("Ask about your Gmail emails... (e.g., 'show yesterday's emails')"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Process with enhanced assistant
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    response_parts = []
                    
                    try:
                        for chunk in assistant.process_request(prompt):
                            response_parts.append(chunk)
                            current_response = "\n".join(response_parts)
                            
                            # Enhanced rendering with HTML support
                            if "email-card" in current_response:
                                message_placeholder.markdown(current_response, unsafe_allow_html=True)
                            elif "🧠" in current_response or "🔍" in current_response:
                                message_placeholder.markdown(f'<div class="tool-response">{current_response}</div>', unsafe_allow_html=True)
                            elif "❌" in current_response:
                                message_placeholder.markdown(f'<div class="error-box">{current_response}</div>', unsafe_allow_html=True)
                            elif "✅" in current_response:
                                message_placeholder.markdown(f'<div class="success-box">{current_response}</div>', unsafe_allow_html=True)
                            else:
                                message_placeholder.markdown(current_response, unsafe_allow_html=True)
                        
                        full_response = "\n".join(response_parts)
                        
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        message_placeholder.markdown(f'<div class="error-box">{error_msg}</div>', unsafe_allow_html=True)
                        full_response = error_msg
                
                # Add to history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.rerun()
        else:
            st.info("👈 Connect to Gmail MCP server in sidebar to start")
            
            # Enhanced example commands
            st.markdown("""
            **📝 Example Commands Once Connected:**
            
            **📧 Email Retrieval:**
            - *"Show me my recent emails"*
            - *"Get emails from yesterday"*
            - *"What emails did I receive 2 days back?"*
            - *"Show me this week's emails"*
            
            **🔍 Advanced Queries:**
            - *"Find emails from john@company.com"*
            - *"Search for emails about 'meeting'"*
            - *"Get emails with attachments from last week"*
            
            **👥 Contact Management:**
            - *"Search people in my contacts"*
            - *"Find contact information for Sarah"*
            """)
    
    with col2:
        st.subheader("📊 Enhanced Dashboard")
        
        if assistant.gmail_server and assistant.gmail_server.connected:
            # Enhanced connection metrics
            st.metric("Gmail Server", "Connected", delta="Online")
            st.metric("Available Tools", len(assistant.gmail_server.server_info.capabilities))
            st.metric("Response Format", "Enhanced Cards")
            
            st.divider()
            
            # Feature highlights
            st.subheader("✨ Enhanced Features")
            
            features = [
                "🎨 Beautiful email cards",
                "🧹 Clean HTML parsing", 
                "📅 Smart date formatting",
                "🏷️ Label badges",
                "📝 Email previews",
                "👤 Clean sender names",
                "📱 Responsive design"
            ]
            
            for feature in features:
                st.write(feature)
            
            st.divider()
            
            # Usage stats
            st.subheader("📈 Usage Stats")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Messages", len(st.session_state.messages))
            with col_b:
                enhanced_mode = "ON" if "email-card" in str(st.session_state.messages) else "Ready"
                st.metric("Enhanced UI", enhanced_mode)
        else:
            st.info("📊 Dashboard will show stats once connected")
        
        st.divider()
        
        # Enhanced example showcase
        st.subheader("🎨 Preview: Enhanced Output")
        
        # Show sample of what enhanced formatting looks like
        sample_preview = """
        <div class="email-card" style="margin: 0.5rem 0;">
            <div class="email-header">
                <div class="email-from">👤 john.doe@company.com</div>
                <div class="email-subject">Meeting Tomorrow</div>
                <div class="email-date">📅 Today at 2:30 PM</div>
            </div>
            <div class="email-preview">
                💬 Hi team, just a reminder about our meeting tomorrow at 10 AM...
            </div>
            <div class="email-labels">
                <span class="label-badge">IMPORTANT</span>
                <span class="label-badge">WORK</span>
            </div>
        </div>
        """
        
        st.markdown("**Sample Email Card:**")
        st.markdown(sample_preview, unsafe_allow_html=True)
        st.caption("🎯 This is how your emails will look!")
    
    # Enhanced Footer
    st.divider()
    
    # Enhanced metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = 1 if (assistant.gmail_server and assistant.gmail_server.connected) else 0
        st.metric("Gmail Status", "Connected" if status else "Disconnected", 
                 delta="Ready" if status else "Offline")
    
    with col2:
        tools = len(assistant.langchain_tools) if assistant.langchain_tools else 0
        st.metric("LangChain Tools", tools)
    
    with col3:
        st.metric("Chat Messages", len(st.session_state.messages))
    
    with col4:
        mode = "Enhanced" if assistant.api_key else "Direct"
        st.metric("Display Mode", mode, delta="Beautiful Cards")
    
    # Enhanced control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if hasattr(assistant, 'memory') and assistant.memory:
                    assistant.memory.clear()
                st.rerun()
    
    with col2:
        if st.button("🔄 Reset Connection", use_container_width=True):
            if assistant.gmail_server:
                try:
                    assistant.remove_gmail_server()
                    st.success("Connection reset")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reset failed: {str(e)}")
    
    with col3:
        if st.button("🎨 View Sample", use_container_width=True):
            sample_message = {
                "role": "assistant", 
                "content": sample_preview
            }
            st.session_state.messages.append({
                "role": "user",
                "content": "Show me a sample email card"
            })
            st.session_state.messages.append(sample_message)
            st.rerun()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def render_enhanced_message(content: str, message_placeholder):
    """Render messages with enhanced formatting"""
    if "email-card" in content:
        message_placeholder.markdown(content, unsafe_allow_html=True)
    elif "🧠" in content or "🔍" in content:
        message_placeholder.markdown(f'<div class="tool-response">{content}</div>', unsafe_allow_html=True)
    elif "❌" in content:
        message_placeholder.markdown(f'<div class="error-box">{content}</div>', unsafe_allow_html=True)
    elif "✅" in content:
        message_placeholder.markdown(f'<div class="success-box">{content}</div>', unsafe_allow_html=True)
    else:
        message_placeholder.markdown(content, unsafe_allow_html=True)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
