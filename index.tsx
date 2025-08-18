

/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */
import { marked } from 'marked';
import { GoogleGenAI, Type, GenerateContentResponse } from '@google/genai';
import JSZip from 'jszip';

// --- Data for SDK Explorer ---
const ASSISTANTS_API_DOCS = `
Assistants
Beta
Build assistants that can call models and use tools to perform tasks.

Get started with the Assistants API

Create assistant
Beta
post
 
https://api.openai.com/v1/assistants
Create an assistant with a model and instructions.
`;

const CODE_INTERPRETER_DOCS = `
Code Interpreter
================

Allow models to write and run Python to solve problems.

The Code Interpreter tool allows models to write and run Python code in a sandboxed environment to solve complex problems in domains like data analysis, coding, and math. Use it for:

*   Processing files with diverse data and formatting
*   Generating files with data and images of graphs
*   Writing and running code iteratively to solve problems—for example, a model that writes code that fails to run can keep rewriting and running that code until it succeeds
*   Boosting visual intelligence in our latest reasoning models (like [o3](/docs/models/o3) and [o4-mini](/docs/models/o4-mini)). The model can use this tool to crop, zoom, rotate, and otherwise process and transform images.
`;

const REMOTE_MCP_DOCS = `
Remote MCP
==========

Allow models to use remote MCP servers to perform tasks.

[Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) is an open protocol that standardizes how applications provide tools and context to LLMs. The MCP tool in the Responses API allows developers to give the model access to tools hosted on **Remote MCP servers**. These are MCP servers maintained by developers and organizations across the internet that expose these tools to MCP clients, like the Responses API.
`;

const COMPUTER_USE_DOCS = `
Computer use
============

Build a computer-using agent that can perform tasks on your behalf.

**Computer use** is a practical application of our [Computer-Using Agent](https://openai.com/index/computer-using-agent/) (CUA) model, \`computer-use-preview\`, which combines the vision capabilities of [GPT-4o](/docs/models/gpt-4o) with advanced reasoning to simulate controlling computer interfaces and performing tasks.

Computer use is available through the [Responses API](/docs/guides/responses-vs-chat-completions). It is not available on Chat Completions.

How it works
------------

The computer use tool operates in a continuous loop. It sends computer actions, like \`click(x,y)\` or \`type(text)\`, which your code executes on a computer or browser environment and then returns screenshots of the outcomes back to the model.

This loop lets you automate many tasks requiring clicking, typing, scrolling, and more. For example, booking a flight, searching for a product, or filling out a form.
`;

const VECTOR_STORES_DOCS = `
Vector stores
Vector stores power semantic search for the Retrieval API and the file_search tool in the Responses and Assistants APIs.

Related guide: File Search

Create vector store
post
 
https://api.openai.com/v1/vector_stores
Create a vector store.

Request body
chunking_strategy
object

Optional
The chunking strategy used to chunk the file(s). If not set, will use the auto strategy. Only applicable if file_ids is non-empty.
`;

const MENUS_API_DOCS = `
Menus API
=========

Create and manage dynamic, interactive menus for your application.

The Menus API allows you to programmatically define the structure and behavior of menus. These menus can be rendered by your front-end application, providing a consistent way to manage navigation, actions, and settings.

Use cases:
* Main application navigation
* Context-sensitive right-click menus
* User settings dropdowns
* Action toolbars

Create menu
-----------
post
https://api.openai.com/v1/menus

Creates a new menu container.

Request body
name
string

Required
A human-readable name for the menu.

description
string

Optional
A short description of the menu's purpose.

Example request:
{
  "name": "Main Navigation Menu",
  "description": "The primary navigation for the marketing website."
}

Add menu item
-------------
post
https://api.openai.com/v1/menus/{menu_id}/items

Adds a new item to a specified menu.

Path parameters
menu_id
string

Required
The ID of the menu to add the item to.

Request body
label
string

Required
The text to display for the menu item.

action_url
string

Optional
The URL to navigate to when the item is clicked.

icon
string

Optional
The name or URL of an icon to display next to the label.

parent_item_id
string

Optional
If this item is a submenu item, specify the ID of the parent item.

Example request:
{
  "label": "Home",
  "action_url": "/",
  "icon": "home"
}
`;


const COMMANDS_JSON = `
{
  "commands": [
    {
      "name": "List Files",
      "command": "ls",
      "description": "Lists files and directories in the current working directory.",
      "target": "shell"
    },
    {
      "name": "Show Current Path",
      "command": "pwd",
      "description": "Prints the current working directory path.",
      "target": "shell"
    },
    {
      "name": "View README",
      "command": "cat README.md",
      "description": "Displays the content of the README.md file.",
      "target": "shell"
    },
    {
      "name": "Create a Directory",
      "command": "mkdir my-new-folder",
      "description": "Creates a new directory named 'my-new-folder'.",
      "target": "shell"
    },
    {
      "name": "Create a new file",
      "command": "touch app.js",
      "description": "Creates a new empty file named 'app.js'.",
      "target": "shell"
    },
    {
      "name": "Remove a file",
      "command": "rm app.js",
      "description": "Removes the file 'app.js'. Careful, this is permanent.",
      "target": "shell"
    },
    {
      "name": "Install Assistants API",
      "command": "gemini install assistants-api",
      "description": "Installs the Assistants API tool to the active container.",
      "target": "gemini-cli"
    },
    {
      "name": "Clear Terminal",
      "command": "clear",
      "description": "Clears all output from the terminal screen.",
      "target": "shell"
    }
  ]
}
`;

const ICONS: Record<string, string> = {
  'web-dev-assistant': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/><path d="m8 18 4 4 4-4"/><path d="m12 2-4 4h8l-4-4Z"/></svg>`,
  'web-dev-studio': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M9 21V9"/></svg>`,
  'crewai-orchestrator': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>`,
  'command-palette': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m4 6 8-4 8 4"/><path d="m4 18 8 4 8-4"/><path d="M12 2v20"/><polyline points="8,10 4,12 8,14"/></svg>`,
  'assistants-api': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>`,
  'code-interpreter': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  'file-explorer': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>`,
  'remote-mcp': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.5 19H9a7 7 0 1 1 6.71-9h1.79a4.5 4.5 0 1 1 0 9Z"/><path d="M22 10.5V12a10 10 0 1 1-5.93-9.14"/></svg>`,
  'codex': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a10 10 0 0 0-3.54 19.54M12 2a10 10 0 0 1 3.54 19.54"/><path d="M12 2v6"/><path d="m3.54 6.46.71.71"/><path d="M2.05 13h6.9"/><path d="m3.54 17.54-.71.71"/><path d="M12 16v6"/><path d="m20.46 17.54-.71-.71"/><path d="m21.95 13h-6.9"/><path d="m20.46 6.46.71.71"/><path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z"/><path d="M12 2a10 10 0 0 0-3.54 19.54A10 10 0 0 0 12 2Z"/></svg>`,
  'computer-use': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="12" rx="2"/><path d="M8 21h8"/><path d="M12 15v6"/><path d="M10.5 6.5 15 11l-2.5 2.5"/></svg>`,
  'vector-stores': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.27 6.96 12 12.01l8.73-5.05"/><path d="M12 22.08V12"/><path d="m10 9-6 3.5"/><path d="m10 15 6 3.5"/><path d="m14 9 6 3.5"/></svg>`,
  'menus-api': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="18" y2="18"/></svg>`,
  'gemini-cli': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4-4L16.5 3.5z"/><path d="m15 5 4 4"/><path d="M7.5 15.5a2.12 2.12 0 0 1-3-3L17 5l4 4L7.5 15.5z"/></svg>`,
  'folder': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"></path></svg>`,
  'folder-plus': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.93a2 2 0 0 1-1.66-.9l-.82-1.2A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"></path><line x1="12" x2="12" y1="10" y2="16"></line><line x1="9" x2="15" y1="13" y2="13"></line></svg>`,
  'file': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>`,
  'file-plus': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="12" x2="12" y1="18" y2="12"></line><line x1="9" x2="15" y1="15" y2="15"></line></svg>`,
  'zip': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M9 8v8"/><path d="M9 12h6l-3 4-3-4h6"/></svg>`,
  'env': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 12v-2M12 18v-2m0-4v-2"/><path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10Z"/><path d="m9 9 6 6"/><path d="m9 15 6-6"/></svg>`,
  'upload': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>`,
  'download': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
  'refresh': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>`,
  'ai-avatar': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>`,
  'user-avatar': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`,
  'globe': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>`,
  'zap': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>`,
  'database': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>`,
  'send': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>`,
  'edit3': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"></path><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"></path></svg>`,
  'trash2': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>`,
  'plus': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>`,
  'copy': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`,
  'check': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`,
  'alert-triangle': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>`,
  'code2': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 16 4-4-4-4"/><path d="m6 8-4 4 4 4"/><path d="m14.5 4-5 16"/></svg>`,
  'agent-lyra': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>`,
  'agent-kara': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 11 12 14 22 4"></polyline><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg>`,
  'agent-cecilia': `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>`,
};

// --- Type Definitions ---
type ModuleType = 'assistant' | 'docs' | 'commands' | 'explorer' | 'studio' | 'orchestrator';

interface Module {
  title: string;
  type: ModuleType;
  content: string | null;
  installableToolName?: string;
}

interface FileNode {
  id: string;
  name: string;
  type: 'file' | 'directory';
  path: string; // Full path from root
  content?: string; // Base64 encoded for binary or string for text
  encoding?: 'utf8' | 'base64';
  children?: FileNode[];
  // New fields for table view
  description?: string;
  author?: string;
  createdAt: number;
  modifiedAt: number;
}

interface Container {
  id: string;
  name: string;
  createdAt: number;
  expiresAt: number;
  files: FileNode[]; // This represents the root directory '/'
  installedTools: string[];
}

interface ContainerTemplate {
    id: string;
    name: string;
    tools: string[];
}

interface Bookmark {
  id: string;
  prompt: string;
  responseHtml: string;
  sources: any[];
  createdAt: number;
}

interface AgentMessage {
  id: string;
  role: 'user' | 'ai';
  htmlContent: string;
}

interface CrewAgent {
    id: 'lyra' | 'kara' | 'sophia' | 'cecilia';
    name: string;
    role: string;
    icon: string;
    promptTemplate: (context: string) => string;
}

// --- Module Configuration ---
const modules: Record<string, Module> = {
  'web-dev-assistant': {
    title: 'WebDev Assistant',
    type: 'assistant',
    content: null,
  },
  'web-dev-studio': {
    title: 'Editor',
    type: 'studio',
    content: null,
  },
  'file-explorer': {
    title: 'File Explorer',
    type: 'explorer',
    content: null,
  },
  'crewai-orchestrator': {
    title: 'CrewAI Orchestrator',
    type: 'orchestrator',
    content: null,
  },
  'command-palette': {
    title: 'Command Palette',
    type: 'commands',
    content: COMMANDS_JSON,
  },
  'assistants-api': {
    title: 'Assistants API',
    type: 'docs',
    content: ASSISTANTS_API_DOCS,
    installableToolName: 'assistants-api',
  },
  'code-interpreter': {
    title: 'Code Interpreter',
    type: 'docs',
    content: CODE_INTERPRETER_DOCS,
    installableToolName: 'code-interpreter',
  },
  'remote-mcp': {
    title: 'Remote MCP',
    type: 'docs',
    content: REMOTE_MCP_DOCS,
    installableToolName: 'remote-mcp',
  },
  'codex': {
    title: 'Codex',
    type: 'docs',
    content: null, // Content is now rendered programmatically
  },
   'computer-use': {
    title: 'Computer Use',
    type: 'docs',
    content: COMPUTER_USE_DOCS,
    installableToolName: 'computer-use',
  },
  'vector-stores': {
    title: 'Vector Stores',
    type: 'docs',
    content: VECTOR_STORES_DOCS,
    installableToolName: 'vector-stores',
  },
  'menus-api': {
    title: 'Menus API',
    type: 'docs',
    content: MENUS_API_DOCS,
    installableToolName: 'menus-api',
  },
};

const CONTAINER_TEMPLATES: ContainerTemplate[] = [
    { id: 'default', name: 'Default Peering', tools: [] },
    { id: 'web-dev', name: 'Web Dev Client', tools: ['assistants-api', 'code-interpreter', 'menus-api'] },
    { id: 'agentic', name: 'Agentic Client', tools: ['computer-use', 'remote-mcp'] },
    { id: 'data-science', name: 'Data Science Client', tools: ['code-interpreter', 'vector-stores'] },
    { id: 'full-suite', name: 'Full Suite', tools: ['assistants-api', 'code-interpreter', 'remote-mcp', 'computer-use', 'vector-stores', 'menus-api'] }
];

const CREW_AGENTS: CrewAgent[] = [
    { 
        id: 'lyra', name: 'LYRA', role: 'Lead Project Manager', icon: ICONS['agent-lyra'], 
        promptTemplate: (task: string) => `As LYRA, a meticulous project manager, your task is to receive a high-level user request and break it down into a clear, sequential plan for your team of AI agents. The plan should outline the steps for validation, API interaction, and reporting. Keep it concise and actionable. User Request: "${task}"`
    },
    { 
        id: 'kara', name: 'KARA', role: 'File Validation Specialist', icon: ICONS['agent-kara'], 
        promptTemplate: (context: string) => `As KARA, a file validation specialist, you've received the project plan. Based on this plan, describe the specific validation checks you will perform. Be detailed about the data formats, required fields, and integrity checks you'll execute. Project Plan: "${context}"`
    },
    { 
        id: 'sophia', name: 'SOPHIA', role: 'API Integration Expert', icon: ICONS['send'], 
        promptTemplate: (context: string) => `As SOPHIA, an API integration expert, your task is to post the validated data. Based on the validation report, describe the structure of the JSON payload you will send to the API endpoint and mention the endpoint you'll be POSTing to. Validation Report: "${context}"`
    },
    { 
        id: 'cecilia', name: 'CECILIA', role: 'Payload Reporting Analyst', icon: ICONS['agent-cecilia'], 
        promptTemplate: (context: string) => `As CECILIA, a payload reporting analyst, you have received the successful API response. Format the key information from the response into a clean, human-readable summary report in markdown format. API Response Details: "${context}"`
    }
];


// --- IndexedDB Helper ---
class DBHelper {
  private db: IDBDatabase | null = null;
  private readonly DB_NAME = 'sdk-explorer-db';
  private readonly CONTAINER_STORE = 'containers';
  private readonly BOOKMARK_STORE = 'bookmarks';
  private readonly KEY_VALUE_STORE = 'keyValueStore';


  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, 3); // Version bump for schema change
      request.onerror = () => reject('Error opening DB');
      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve();
      };
      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.CONTAINER_STORE)) {
          db.createObjectStore(this.CONTAINER_STORE, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(this.BOOKMARK_STORE)) {
          db.createObjectStore(this.BOOKMARK_STORE, { keyPath: 'id' });
        }
        if (!db.objectStoreNames.contains(this.KEY_VALUE_STORE)) {
          db.createObjectStore(this.KEY_VALUE_STORE, { keyPath: 'key' });
        }
      };
    });
  }

  private getStore(storeName: string, mode: IDBTransactionMode): IDBObjectStore {
    if (!this.db) throw new Error('DB not initialized');
    return this.db.transaction(storeName, mode).objectStore(storeName);
  }

  async addContainer(container: Container): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.CONTAINER_STORE, 'readwrite').add(container);
      request.onsuccess = () => resolve();
      request.onerror = () => reject('Failed to add container');
    });
  }

  async getContainer(id: string): Promise<Container | undefined> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.CONTAINER_STORE, 'readonly').get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject('Failed to get container');
    });
  }
  
  async getAllContainers(): Promise<Container[]> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.CONTAINER_STORE, 'readonly').getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject('Failed to get all containers');
    });
  }

  async updateContainer(container: Container): Promise<void> {
     return new Promise((resolve, reject) => {
        const request = this.getStore(this.CONTAINER_STORE, 'readwrite').put(container);
        request.onsuccess = () => resolve();
        request.onerror = () => reject('Failed to update container');
    });
  }

  async deleteContainer(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.CONTAINER_STORE, 'readwrite').delete(id);
      request.onsuccess = () => resolve();
      request.onerror = () => reject('Failed to delete container');
    });
  }

  async addBookmark(bookmark: Bookmark): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.BOOKMARK_STORE, 'readwrite').add(bookmark);
      request.onsuccess = () => resolve();
      request.onerror = () => reject('Failed to add bookmark');
    });
  }

  async getBookmark(id: string): Promise<Bookmark | undefined> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.BOOKMARK_STORE, 'readonly').get(id);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject('Failed to get bookmark');
    });
  }

  async getAllBookmarks(): Promise<Bookmark[]> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.BOOKMARK_STORE, 'readonly').getAll();
      request.onsuccess = () => resolve(request.result.sort((a,b) => b.createdAt - a.createdAt));
      request.onerror = () => reject('Failed to get all bookmarks');
    });
  }

  async deleteBookmark(id: string): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.BOOKMARK_STORE, 'readwrite').delete(id);
      request.onsuccess = () => resolve();
      request.onerror = () => reject('Failed to delete bookmark');
    });
  }

  async saveValue(key: string, value: any): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.KEY_VALUE_STORE, 'readwrite').put({ key, value });
      request.onsuccess = () => resolve();
      request.onerror = () => reject(`Failed to save value for key: ${key}`);
    });
  }
  
  async getValue(key: string): Promise<any | undefined> {
    return new Promise((resolve, reject) => {
      const request = this.getStore(this.KEY_VALUE_STORE, 'readonly').get(key);
      request.onsuccess = () => resolve(request.result ? request.result.value : undefined);
      request.onerror = () => reject(`Failed to get value for key: ${key}`);
    });
  }
}

// --- App Initialization ---
document.addEventListener('DOMContentLoaded', async () => {
  // WebDev Assistant Elements
  const form = document.getElementById('prompt-form') as HTMLFormElement;
  const input = document.getElementById('prompt-input') as HTMLInputElement;
  const submitButton = document.getElementById('submit-button') as HTMLButtonElement;
  const spinner = document.getElementById('spinner') as HTMLDivElement;
  const responseContent = document.getElementById('response-content') as HTMLDivElement;
  const sourcesContainer = document.getElementById('sources-container') as HTMLDivElement;
  const resultContainer = document.getElementById('result-container') as HTMLElement;
  
  // Editor (Web Dev Studio) Elements
  const studioContainerStatus = document.getElementById('studio-container-status') as HTMLParagraphElement;
  const previewFrame = document.getElementById('preview-frame') as HTMLIFrameElement;
  const studioFileToolbar = document.getElementById('studio-file-toolbar') as HTMLDivElement;
  const studioCodeEditor = document.getElementById('studio-code-editor') as HTMLTextAreaElement;
  const previewWrapper = document.getElementById('preview-wrapper') as HTMLDivElement;
  const editorTabBtn = document.getElementById('editor-tab-btn') as HTMLButtonElement;
  const previewTabBtn = document.getElementById('preview-tab-btn') as HTMLButtonElement;
  const studioActiveFileName = document.getElementById('studio-active-file-name') as HTMLSpanElement;
  const studioFileList = document.getElementById('studio-file-list') as HTMLUListElement;

  // CrewAI Orchestrator Elements
  const crewForm = document.getElementById('crew-form') as HTMLFormElement;
  const crewInput = document.getElementById('crew-input') as HTMLTextAreaElement;
  const crewRunButton = document.getElementById('crew-run-button') as HTMLButtonElement;
  const crewSpinner = document.getElementById('crew-spinner') as HTMLDivElement;
  const crewProgressPanel = document.getElementById('crew-progress-panel') as HTMLElement;

  // Layout Elements
  const sidebar = document.getElementById('sidebar') as HTMLElement;
  const resizeHandle = document.getElementById('resize-handle') as HTMLDivElement;
  const collapseBtn = document.getElementById('collapse-btn') as HTMLButtonElement;

  // Page Elements
  const navLinksContainer = document.getElementById('nav-links') as HTMLUListElement;
  const allPages = document.querySelectorAll('.page') as NodeListOf<HTMLDivElement>;
  const webDevAssistantPage = document.getElementById('web-dev-assistant-page') as HTMLDivElement;
  const webDevStudioPage = document.getElementById('web-dev-studio-page') as HTMLDivElement;
  const fileExplorerPage = document.getElementById('file-explorer-page') as HTMLDivElement;
  const crewaiOrchestratorPage = document.getElementById('crewai-orchestrator-page') as HTMLDivElement;
  const docsPage = document.getElementById('docs-page') as HTMLDivElement;
  const docsHeader = document.getElementById('docs-header') as HTMLDivElement;
  const docsContent = document.getElementById('docs-content') as HTMLDivElement;
  const commandsPage = document.getElementById('commands-page') as HTMLDivElement;
  const commandsTitle = document.getElementById('commands-title') as HTMLHeadingElement;
  const commandsContent = document.getElementById('commands-content') as HTMLDivElement;

  // Bookmarks Elements
  const bookmarksPanel = document.getElementById('bookmarks-panel') as HTMLElement;
  const toggleBookmarksBtn = document.getElementById('toggle-bookmarks-btn') as HTMLButtonElement;
  const closeBookmarksBtn = document.getElementById('close-bookmarks-btn') as HTMLButtonElement;
  const bookmarksList = document.getElementById('bookmarks-list') as HTMLUListElement;
  const bookmarksCount = document.getElementById('bookmarks-count') as HTMLSpanElement;
  
  // File Explorer Elements
  const createContainerForm = document.getElementById('create-container-form') as HTMLFormElement;
  const containerNameInput = document.getElementById('container-name-input') as HTMLInputElement;
  const containerTemplateSelect = document.getElementById('container-template-select') as HTMLSelectElement;
  const containersList = document.getElementById('containers-list') as HTMLUListElement;
  const fileExplorerList = document.getElementById('file-explorer-list') as HTMLUListElement;
  const fileExplorerTitle = document.getElementById('file-explorer-title') as HTMLHeadingElement;
  const fileExplorerToolbar = document.getElementById('file-explorer-toolbar') as HTMLDivElement;
  const installedToolsList = document.getElementById('installed-tools-list') as HTMLUListElement;

  // Terminal Elements
  const terminalConsole = document.getElementById('terminal-console') as HTMLElement;
  const toggleTerminalBtn = document.getElementById('toggle-terminal-btn') as HTMLButtonElement;
  const terminalOutput = document.getElementById('terminal-output') as HTMLDivElement;
  const terminalForm = document.getElementById('terminal-form') as HTMLFormElement;
  const terminalInput = document.getElementById('terminal-input') as HTMLInputElement;
  const terminalPathSpan = document.getElementById('terminal-path') as HTMLSpanElement;

  // --- State Management ---
  const db = new DBHelper();
  await db.init();

  let activeContainerId: string | null = await db.getValue('activeContainerId');
  let activeContainer: Container | null = null;
  let activeFileId: string | null = null;
  let bookmarks: Bookmark[] = [];
  let isSidebarCollapsed = await db.getValue('sidebarCollapsed') || false;
  let isBookmarksPanelCollapsed = true;
  let isTerminalCollapsed = true;
  let currentPath = '/';
  
  // Gemini AI Client
  let ai: GoogleGenAI;
  try {
    ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  } catch (error) {
    console.error("Failed to initialize GoogleGenAI:", error);
    // You might want to display an error to the user here
  }

  // --- Helper Functions ---
  const generateId = () => `id-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const formatDate = (timestamp: number) => new Date(timestamp).toLocaleString();
  const base64ToUtf8 = (base64: string) => decodeURIComponent(atob(base64).split('').map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2)).join(''));
  const utf8ToBase64 = (str: string) => btoa(encodeURIComponent(str).replace(/%([0-9A-F]{2})/g, (match, p1) => String.fromCharCode(parseInt(p1, 16))));

  const findNodeAndParentById = (id: string, root: FileNode): { node: FileNode | null; parent: FileNode | null } => {
    const queue: { node: FileNode; parent: FileNode | null }[] = [{ node: root, parent: null }];
    while (queue.length > 0) {
        const { node, parent } = queue.shift()!;
        if (node.id === id) return { node, parent };
        if (node.children) {
            for (const child of node.children) {
                queue.push({ node: child, parent: node });
            }
        }
    }
    return { node: null, parent: null };
  };

  const findNodeByPath = (path: string, root: FileNode): FileNode | null => {
    if (path === '/') return root;
    const parts = path.split('/').filter(p => p);
    let currentNode: FileNode | undefined = root;
    for (const part of parts) {
        if (!currentNode?.children) return null;
        currentNode = currentNode.children.find(c => c.name === part);
        if (!currentNode) return null;
    }
    return currentNode || null;
  };
  
  // --- UI Rendering ---

  function populateNavLinks() {
    navLinksContainer.innerHTML = '';
    const groupedModules = {
      'Main': ['web-dev-assistant', 'web-dev-studio', 'file-explorer', 'crewai-orchestrator', 'command-palette'],
      'APIs': ['assistants-api', 'code-interpreter', 'remote-mcp', 'computer-use', 'vector-stores', 'menus-api']
    };

    Object.entries(groupedModules).forEach(([groupName, moduleIds]) => {
        const blockquote = document.createElement('blockquote');
        blockquote.textContent = groupName;
        navLinksContainer.appendChild(blockquote);

        moduleIds.forEach(id => {
            const module = modules[id];
            if (!module) return;
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = `#${id}`;
            a.dataset.moduleId = id;
            a.innerHTML = `${ICONS[id] || ''} <span>${module.title}</span>`;
            li.appendChild(a);
            navLinksContainer.appendChild(li);
        });
    });
  }

  async function renderContainersList() {
    const containers = await db.getAllContainers();
    containersList.innerHTML = containers.map(c => `
      <li data-container-id="${c.id}" class="${c.id === activeContainerId ? 'active' : ''}">
        <blockquote>
          <div>
            <strong>${c.name}</strong><br>
            <small>Created: ${formatDate(c.createdAt)}</small>
          </div>
        </blockquote>
      </li>
    `).join('');
  }

  function renderFileList(directory: FileNode, element: HTMLElement) {
      if (!directory.children || directory.children.length === 0) {
          element.innerHTML = '<li class="empty-state"><blockquote>This directory is empty.</blockquote></li>';
          return;
      }
      // Sort: folders first, then by name
      const sortedChildren = [...directory.children].sort((a, b) => {
          if (a.type === 'directory' && b.type === 'file') return -1;
          if (a.type === 'file' && b.type === 'directory') return 1;
          return a.name.localeCompare(b.name);
      });

      element.innerHTML = sortedChildren.map(node => {
        const icon = ICONS[node.type];
        return `
          <li data-file-id="${node.id}" data-file-path="${node.path}">
            <blockquote>
              <div class="file-item-main">
                ${icon}
                <span>${node.name}</span>
              </div>
              <div class="file-actions">
                ${node.name.endsWith('.env') ? `<button title="Edit Env" class="edit-env-btn">${ICONS['edit3']}</button>` : ''}
                <button title="Delete" class="delete-file-btn">${ICONS['trash2']}</button>
              </div>
            </blockquote>
          </li>
        `;
      }).join('');
  }
  
  function updateExplorerAndStudioViews() {
      if (!activeContainer) {
          fileExplorerList.innerHTML = '<li class="empty-state"><blockquote>Select a container.</blockquote></li>';
          studioFileList.innerHTML = '<li class="empty-state"><blockquote>Select a container.</blockquote></li>';
          fileExplorerTitle.textContent = 'Select a Container';
          studioContainerStatus.textContent = 'No active container.';
          installedToolsList.innerHTML = '';
          fileExplorerToolbar.innerHTML = '';
          studioFileToolbar.innerHTML = '';
          terminalInput.disabled = true;
          return;
      }
      
      const rootNode: FileNode = {
          id: 'root', name: '/', type: 'directory', path: '/', createdAt: 0, modifiedAt: 0,
          children: activeContainer.files
      };
      
      const currentNode = findNodeByPath(currentPath, rootNode) || rootNode;
      
      renderFileList(currentNode, fileExplorerList);
      renderFileList(rootNode, studioFileList); // Studio always shows root
      
      fileExplorerTitle.textContent = activeContainer.name;
      studioContainerStatus.textContent = `Active: ${activeContainer.name}`;
      terminalInput.disabled = false;

      // Render toolbars
      const toolbarButtons = `
        <button id="add-file-btn" title="New File">${ICONS['file-plus']}</button>
        <button id="add-folder-btn" title="New Folder">${ICONS['folder-plus']}</button>
        <button id="upload-file-btn" title="Upload Files">${ICONS['upload']}</button>
        <button id="download-zip-btn" title="Download as ZIP">${ICONS['download']}</button>
        <button id="refresh-files-btn" title="Refresh">${ICONS['refresh']}</button>
      `;
      fileExplorerToolbar.innerHTML = toolbarButtons;
      studioFileToolbar.innerHTML = toolbarButtons;

      // Render installed tools
      installedToolsList.innerHTML = activeContainer.installedTools.map(toolId => {
          const module = modules[toolId];
          if (!module) return '';
          return `
            <li>
              <blockquote>
                ${ICONS[toolId]}
                <span>${module.title}</span>
              </blockquote>
            </li>
          `;
      }).join('') || '<li><blockquote>No tools installed.</blockquote></li>';
  }


  function renderDocsPage(moduleId: string) {
    const module = modules[moduleId];
    if (!module || !module.content) {
      docsContent.innerHTML = '<p>Documentation not available.</p>';
      return;
    }

    let installButton = '';
    if (module.installableToolName) {
        const isInstalled = activeContainer?.installedTools.includes(module.installableToolName);
        installButton = `<button id="install-tool-btn" data-tool="${module.installableToolName}" ${!activeContainer || isInstalled ? 'disabled' : ''}>${isInstalled ? 'Installed' : 'Install Tool'}</button>`
    }

    docsHeader.innerHTML = `
        <h1 id="docs-title">${module.title}</h1>
        ${installButton}
    `;
    docsContent.innerHTML = marked.parse(module.content);
  }

  function renderCommandsPage() {
    const data = JSON.parse(COMMANDS_JSON);
    commandsTitle.textContent = "Command Palette";
    commandsContent.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Command</th>
            <th>Description</th>
            <th>Target</th>
          </tr>
        </thead>
        <tbody>
          ${data.commands.map((cmd: any) => `
            <tr data-command='${JSON.stringify(cmd)}'>
              <td>${cmd.name}</td>
              <td><code>${cmd.command}</code></td>
              <td>${cmd.description}</td>
              <td>${cmd.target}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  }
  
  async function updateBookmarks() {
    bookmarks = await db.getAllBookmarks();
    bookmarksCount.textContent = bookmarks.length.toString();
    bookmarksList.innerHTML = bookmarks.map(b => `
      <li class="bookmark-item" data-bookmark-id="${b.id}">
        <blockquote>
          <div class="bookmark-item-header">
            <p>${b.prompt}</p>
          </div>
          <div class="bookmark-item-body">
            ${b.responseHtml}
          </div>
          <div class="bookmark-item-actions">
            <button class="run-btn">Run Again</button>
            <button class="delete-btn">Delete</button>
          </div>
        </blockquote>
      </li>
    `).join('');
  }

  function renderCrewOrchestrator() {
    const agentMarkup = CREW_AGENTS.map(agent => `
      <div class="agent-step" id="agent-step-${agent.id}">
        <div class="agent-step-header">
          <div class="agent-avatar">${agent.icon}</div>
          <div class="agent-details">
            <h3>${agent.name}</h3>
            <p>${agent.role}</p>
          </div>
          <div class="agent-status pending">Pending</div>
        </div>
        <div class="agent-output"></div>
      </div>
    `).join('');
    crewProgressPanel.innerHTML = agentMarkup;
  }
  
  // --- Navigation ---
  function navigateTo(moduleId: string) {
    allPages.forEach(p => p.classList.remove('active'));
    document.querySelectorAll('#nav-links a').forEach(a => a.classList.remove('active'));

    const pageId = `${moduleId}-page`;
    let page = document.getElementById(pageId);
    const module = modules[moduleId];
    
    if (module.type === 'docs') {
      page = docsPage;
      renderDocsPage(moduleId);
    } else if (module.type === 'commands') {
      page = commandsPage;
      renderCommandsPage();
    }
    
    if (page) {
      page.classList.add('active');
      const link = document.querySelector(`#nav-links a[data-module-id="${moduleId}"]`);
      if (link) {
        link.classList.add('active');
      }
    }

    window.location.hash = moduleId;
    db.saveValue('lastPage', moduleId);
  }

  // --- Event Handlers ---
  function handleNavLinkClick(e: Event) {
    const target = e.target as HTMLElement;
    const link = target.closest('a');
    if (link && link.dataset.moduleId) {
      e.preventDefault();
      navigateTo(link.dataset.moduleId);
    }
  }

  async function handleAssistantFormSubmit(e: Event) {
    e.preventDefault();
    if (!ai) {
        responseContent.innerHTML = '<p class="error">Gemini API not initialized.</p>';
        resultContainer.style.display = 'block';
        return;
    }
    const prompt = input.value.trim();
    if (!prompt) return;

    submitButton.disabled = true;
    spinner.style.display = 'block';
    responseContent.innerHTML = '';
    sourcesContainer.innerHTML = '';
    resultContainer.style.display = 'block';

    try {
      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: { tools: [{googleSearch: {}}] },
      });

      const text = response.text;
      const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks || [];
      const sources = groundingChunks
          .map((chunk: any) => chunk.web)
          .filter((web: any) => web && web.uri);

      const responseHtml = marked.parse(text);
      responseContent.innerHTML = responseHtml;
      
      if (sources.length > 0) {
        sourcesContainer.innerHTML = `
          <h2>Sources</h2>
          <ul>
            ${sources.map((s: any) => `<li><a href="${s.uri}" target="_blank">${s.title || s.uri}</a></li>`).join('')}
          </ul>
        `;
      }
      
      const actionsContainer = document.createElement('div');
      actionsContainer.className = 'result-actions';
      actionsContainer.innerHTML = `
        <button id="bookmark-btn" title="Bookmark this response">${ICONS['plus']} <span>Bookmark</span></button>
        <button id="copy-response-btn" title="Copy response text">${ICONS['copy']} <span>Copy</span></button>
      `;
      resultContainer.prepend(actionsContainer);

      document.getElementById('bookmark-btn')?.addEventListener('click', async () => {
        const newBookmark: Bookmark = {
          id: generateId(),
          prompt: prompt,
          responseHtml: responseHtml,
          sources: sources,
          createdAt: Date.now()
        };
        await db.addBookmark(newBookmark);
        updateBookmarks();
      });

      document.getElementById('copy-response-btn')?.addEventListener('click', () => {
        navigator.clipboard.writeText(text);
      });

    } catch (error) {
      console.error('Gemini API Error:', error);
      responseContent.innerHTML = `<p class="error">An error occurred. Please check the console for details.</p>`;
    } finally {
      submitButton.disabled = false;
      spinner.style.display = 'none';
    }
  }

  async function handleCrewFormSubmit(e: Event) {
    e.preventDefault();
    if (!ai) return;

    const task = crewInput.value.trim();
    if (!task) return;

    crewRunButton.disabled = true;
    crewSpinner.style.display = 'block';
    renderCrewOrchestrator(); // Reset panel
    
    let context = task;

    for (const agent of CREW_AGENTS) {
        const stepElement = document.getElementById(`agent-step-${agent.id}`)!;
        const statusElement = stepElement.querySelector('.agent-status')!;
        const outputElement = stepElement.querySelector('.agent-output')!;

        stepElement.classList.add('working');
        statusElement.textContent = 'Working';
        statusElement.className = 'agent-status working';

        try {
            const prompt = agent.promptTemplate(context);
            const response: GenerateContentResponse = await ai.models.generateContent({
              model: 'gemini-2.5-flash',
              contents: prompt
            });
            const resultText = response.text;
            context = resultText; // Pass the output as context to the next agent
            
            outputElement.innerHTML = marked.parse(resultText);
            (outputElement as HTMLElement).style.display = 'block';

            stepElement.classList.remove('working');
            stepElement.classList.add('completed');
            statusElement.textContent = 'Completed';
            statusElement.className = 'agent-status completed';
        } catch (error) {
            console.error(`Error with agent ${agent.name}:`, error);
            outputElement.innerHTML = '<p class="error">An error occurred.</p>';
            stepElement.classList.remove('working');
            statusElement.textContent = 'Failed';
            break; // Stop the crew on error
        }
    }

    crewRunButton.disabled = false;
    crewSpinner.style.display = 'none';
  }

  // --- Sidebar and Layout Handlers ---
  function handleSidebarCollapse() {
    isSidebarCollapsed = !isSidebarCollapsed;
    sidebar.classList.toggle('collapsed', isSidebarCollapsed);
    db.saveValue('sidebarCollapsed', isSidebarCollapsed);
  }

  function handleSidebarResize(e: MouseEvent) {
    const startX = e.clientX;
    const startWidth = sidebar.offsetWidth;
    const handleMouseMove = (e: MouseEvent) => {
      const newWidth = startWidth + (e.clientX - startX);
      if (newWidth > 200 && newWidth < 500) {
        sidebar.style.width = `${newWidth}px`;
      }
    };
    const handleMouseUp = () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }

  function toggleBookmarksPanel() {
    isBookmarksPanelCollapsed = !isBookmarksPanelCollapsed;
    bookmarksPanel.classList.toggle('collapsed', isBookmarksPanelCollapsed);
  }

  function toggleTerminal() {
    isTerminalCollapsed = !isTerminalCollapsed;
    terminalConsole.classList.toggle('collapsed', isTerminalCollapsed);
    document.body.classList.toggle('terminal-collapsed', isTerminalCollapsed);
  }

  // --- File Explorer and Studio Logic ---
  async function loadContainer(id: string) {
      activeContainer = await db.getContainer(id);
      activeContainerId = id;
      await db.saveValue('activeContainerId', id);
      currentPath = '/';
      terminalPathSpan.textContent = `${activeContainer?.name || '...'} ~${currentPath}`;
      updateExplorerAndStudioViews();
  }

  async function handleCreateContainer(e: Event) {
      e.preventDefault();
      const name = containerNameInput.value.trim();
      if (!name) return;
      
      const templateId = containerTemplateSelect.value;
      const template = CONTAINER_TEMPLATES.find(t => t.id === templateId)!;

      const newContainer: Container = {
          id: generateId(),
          name,
          createdAt: Date.now(),
          expiresAt: Date.now() + 7 * 24 * 60 * 60 * 1000, // 7 days
          files: [],
          installedTools: template.tools,
      };
      await db.addContainer(newContainer);
      containerNameInput.value = '';
      await renderContainersList();
      await loadContainer(newContainer.id);
  }

  // --- Terminal Logic ---
  function appendToTerminal(html: string, type: 'command' | 'output' | 'error' | 'info' | 'success') {
      const line = document.createElement('div');
      line.className = `terminal-line ${type}`;
      line.innerHTML = html;
      terminalOutput.appendChild(line);
      terminalOutput.scrollTop = terminalOutput.scrollHeight;
  }

  function handleTerminalCommand(e: Event) {
      e.preventDefault();
      if (!activeContainer) return;

      const command = terminalInput.value.trim();
      if (!command) return;

      appendToTerminal(`<span class="prompt-char">$</span> <span class="command-text">${command}</span>`, 'command');
      
      const [cmd, ...args] = command.split(' ');
      
      switch(cmd) {
          case 'clear':
              terminalOutput.innerHTML = '';
              break;
          case 'ls':
              const rootNode: FileNode = { id: 'root', name: '/', type: 'directory', path: '/', createdAt: 0, modifiedAt: 0, children: activeContainer.files };
              const targetNode = findNodeByPath(currentPath, rootNode);
              if (targetNode && targetNode.children) {
                  appendToTerminal(targetNode.children.map(c => c.name).join('\n'), 'output');
              } else {
                  appendToTerminal('Directory not found or empty.', 'error');
              }
              break;
          case 'pwd':
              appendToTerminal(currentPath, 'output');
              break;
          case 'gemini':
              if (args[0] === 'install' && args[1]) {
                  const toolName = args[1];
                  if (modules[toolName] && modules[toolName].installableToolName) {
                      if (!activeContainer.installedTools.includes(toolName)) {
                          activeContainer.installedTools.push(toolName);
                          db.updateContainer(activeContainer);
                          updateExplorerAndStudioViews();
                          appendToTerminal(`Tool '${toolName}' installed successfully.`, 'success');
                      } else {
                          appendToTerminal(`Tool '${toolName}' is already installed.`, 'info');
                      }
                  } else {
                      appendToTerminal(`Unknown tool: ${toolName}`, 'error');
                  }
              }
              break;
          default:
              appendToTerminal(`Command not found: ${cmd}`, 'error');
      }

      terminalInput.value = '';
  }

  // --- Initial Setup ---
  async function initializeApp() {
    // Layout
    if (isSidebarCollapsed) sidebar.classList.add('collapsed');
    if (isTerminalCollapsed) {
        terminalConsole.classList.add('collapsed');
        document.body.classList.add('terminal-collapsed');
    }
    bookmarksPanel.classList.add('collapsed');

    // Navigation
    populateNavLinks();
    navLinksContainer.addEventListener('click', handleNavLinkClick);
    
    // Set initial page
    const lastPage = await db.getValue('lastPage');
    const initialModuleId = window.location.hash.substring(1) || lastPage || 'web-dev-assistant';
    navigateTo(initialModuleId);

    // Sidebar
    collapseBtn.addEventListener('click', handleSidebarCollapse);
    resizeHandle.addEventListener('mousedown', handleSidebarResize);

    // Assistant
    form.addEventListener('submit', handleAssistantFormSubmit);

    // Bookmarks
    await updateBookmarks();
    toggleBookmarksBtn.addEventListener('click', toggleBookmarksPanel);
    closeBookmarksBtn.addEventListener('click', toggleBookmarksPanel);
    bookmarksList.addEventListener('click', async (e) => {
        const target = e.target as HTMLElement;
        const item = target.closest('.bookmark-item');
        if (!item) return;

        const bookmarkId = item.dataset.bookmarkId!;
        const bookmark = await db.getBookmark(bookmarkId);
        if (!bookmark) return;
        
        if (target.closest('.bookmark-item-header')) {
            item.querySelector('.bookmark-item-body')?.classList.toggle('expanded');
        } else if (target.classList.contains('run-btn')) {
            input.value = bookmark.prompt;
            navigateTo('web-dev-assistant');
            form.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
        } else if (target.classList.contains('delete-btn')) {
            await db.deleteBookmark(bookmarkId);
            await updateBookmarks();
        }
    });
    
    // File Explorer
    CONTAINER_TEMPLATES.forEach(t => {
        const option = document.createElement('option');
        option.value = t.id;
        option.textContent = t.name;
        containerTemplateSelect.appendChild(option);
    });
    await renderContainersList();
    if(activeContainerId) {
        await loadContainer(activeContainerId);
    } else {
        updateExplorerAndStudioViews();
    }
    createContainerForm.addEventListener('submit', handleCreateContainer);
    containersList.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const li = target.closest('li');
        if (li && li.dataset.containerId) {
            loadContainer(li.dataset.containerId);
            renderContainersList(); // To update active class
        }
    });

    // CrewAI Orchestrator
    crewForm.addEventListener('submit', handleCrewFormSubmit);
    renderCrewOrchestrator();

    // Terminal
    toggleTerminalBtn.addEventListener('click', toggleTerminal);
    terminalForm.addEventListener('submit', handleTerminalCommand);

    // Editor / Studio
    editorTabBtn.addEventListener('click', () => {
        editorTabBtn.classList.add('active');
        previewTabBtn.classList.remove('active');
        studioCodeEditor.classList.remove('hidden');
        previewWrapper.classList.add('hidden');
    });

    previewTabBtn.addEventListener('click', () => {
        previewTabBtn.classList.add('active');
        editorTabBtn.classList.remove('active');
        previewWrapper.classList.remove('hidden');
        studioCodeEditor.classList.add('hidden');
    });
    
    studioCodeEditor.addEventListener('change', async () => {
        if (!activeContainer || !activeFileId) return;
        const root: FileNode = {id: 'root-dummy', name: '/', type:'directory', path: '/', createdAt:0, modifiedAt:0, children: activeContainer.files};
        const { node } = findNodeAndParentById(activeFileId, root);
        if (node && node.type === 'file') {
            node.content = utf8ToBase64(studioCodeEditor.value);
            node.modifiedAt = Date.now();
            await db.updateContainer(activeContainer);
            if (node.name.endsWith('.html')) {
                previewFrame.srcdoc = studioCodeEditor.value;
            }
        }
    });

    [fileExplorerList, studioFileList].forEach(list => {
      list.addEventListener('click', (e) => {
        const li = (e.target as HTMLElement).closest('li');
        if (!li || !li.dataset.fileId || !activeContainer) return;
        const fileId = li.dataset.fileId;
        const root: FileNode = {id: 'root-dummy', name: '/', type:'directory', path: '/', createdAt:0, modifiedAt:0, children: activeContainer.files};
        const { node } = findNodeAndParentById(fileId, root);

        if (node) {
          if (node.type === 'file') {
            activeFileId = fileId;
            studioActiveFileName.textContent = node.name;
            const content = node.content ? base64ToUtf8(node.content) : '';
            studioCodeEditor.value = content;
            if (node.name.endsWith('.html')) {
              previewFrame.srcdoc = content;
            } else {
              previewFrame.srcdoc = '';
            }
            navigateTo('web-dev-studio');
          } else if (node.type === 'directory') {
            // Handle directory navigation for file explorer
            if (list === fileExplorerList) {
              currentPath = node.path;
              terminalPathSpan.textContent = `${activeContainer?.name || '...'} ~${currentPath}`;
              updateExplorerAndStudioViews();
            }
          }
        }
      });
    });
  }
  
  await initializeApp();

});