from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import ollama
from openai import OpenAI
import duckdb
from trino.dbapi import connect

class ParameterDef(BaseModel):
    type: str = 'string'
    description: str = ''

class FunctionParameters(BaseModel):
    type: str
    properties: Dict[str, ParameterDef]
    required: list[str] = []

class FunctionInnerDef(BaseModel):
    name: str
    description: str
    parameters: FunctionParameters

class FunctionDef(BaseModel):
    type: str
    function: FunctionInnerDef

class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCallRequest(BaseModel):
    id: str
    type: str
    function: FunctionCall

class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None

class ChatApiRequest(BaseModel):
    openaiApiKey: str | None = None
    model: str
    messages: list[ChatMessage]
    tools: list[ollama.Tool] = []
    tool_choice: str = 'auto'

class ChatResponseChoice(BaseModel):
    message: ChatMessage
    index: int
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatResponseChoice]

class ExecuteSqlRequest(BaseModel):
    code: str
    redshiftCredentials: dict | None = None

class ExecuteSqlResponse(BaseModel):
    columns: list[str]
    types: list[str]
    rows: list
    error: str | None = None
    connectionTested: bool | None = None
    query: str | None = None


app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client=OpenAI()

@app.post("/api/chat")
async def chat(message: ChatApiRequest) -> ChatResponse:
    print(f"/api/chat: {message.json()}")
    resp = client.chat.completions.create(
        model=message.model,
        messages=message.messages,
        tools=message.tools,
        tool_choice='auto')
    print(f" response: {resp}")
    return resp

# initalize meteorite table
duckdb.sql("""
CREATE OR REPLACE TABLE meteorite AS
  FROM read_csv("./meteorite.csv", header=True,
    columns={
      'name':'varchar',
      'id':'integer',
      'nametype':'varchar',
      'recclass':'varchar',
      'mass (g)':'double',
      'fall':'varchar',
      'year':'integer',
      'reclat':'double',
      'reclong':'double',
      'GeoLocation':'varchar'
    })
""")

def convertToRows(cols, tuples):
    rows = []
    for t in tuples:
        row = {}
        for i in range(len(cols)):
            row[cols[i]] = t[i]
        rows.append(row)
    return rows

@app.post('/api/query')
async def executeQuery(message: ExecuteSqlRequest):
    print(f"/api/query: {message}")
    dbid = message.redshiftCredentials['id']
    if dbid == 'TRINO-RISK':
        # Trino
        try:
            conn = connect(host='localhost', port=8080, user='admin', catalog='memory', schema='default')
            cur = conn.cursor()
            res = cur.execute(message.code)
            columns = [cd.name for cd in cur.description]
            return {
                "columns": columns,
                "types": [cd.type_code for cd in cur.description],
                "query": message.code,
                "rows": convertToRows(columns, cur.fetchall()),
                "error": None,
                "connectionTested": True
            }
        except Exception as e:
            return {
                "columns": None,
                "types": None,
                "query": message.code,
                "rows": None,
                "error": str(e),
                "connectionTested": False
            }
    else:
        # DuckDB
        try:
            res = duckdb.sql(message.code)
            return {
                "columns": res.columns,
                "types": ['any' for _ in res.columns],
                "query": message.code,
                "rows": convertToRows(res.columns, res.fetchall()),
                "error": None,
                "connectionTested": True
            }
        except Exception as e:
            return {
                "columns": None,
                "types": None,
                "query": message.code,
                "rows": None,
                "error": str(e),
                "connectionTested": False
            }

app.mount("/", StaticFiles(directory="dist"), name="dist")
