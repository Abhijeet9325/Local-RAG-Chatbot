from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

model = ChatOllama(model="llama3.2:3b")
parser = StrOutputParser()

print("=" * 60)
print("       🎬 MOVIE INFORMATION EXTRACTOR — PROMPT TEMPLATES")
print("=" * 60)


# ─────────────────────────────────────────────────────────────
# 1. Basic PromptTemplate — Extract basic movie info
# ─────────────────────────────────────────────────────────────
print("\n[1] Basic PromptTemplate — Movie Info Extractor")
print("-" * 40)

basic_template = PromptTemplate(
    input_variables=["movie_name"],
    template="""
    You are a professional movie information extractor.
    Extract and return the following details about the movie: {movie_name}

    Return in this format:
    - Title        :
    - Director     :
    - Release Year :
    - Genre        :
    - IMDb Rating  :
    - Cast         :
    - Plot Summary :
    """
)

prompt = basic_template.format(movie_name="Inception")
response = model.invoke(prompt)
print(response.content)


# ─────────────────────────────────────────────────────────────
# 2. ChatPromptTemplate — Deep movie analyst
# ─────────────────────────────────────────────────────────────
print("\n[2] ChatPromptTemplate — Deep Movie Analyst")
print("-" * 40)

chat_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are a world-class movie information extractor and analyst.
        Your job is to extract deep insights about movies.
        Always structure your response clearly.
        Focus area: {focus_area}
     """),
    ("human", "Extract information about the movie: {movie_name}")
])

chain = chat_template | model | parser

response = chain.invoke({
    "focus_area": "cinematography, themes, and symbolism",
    "movie_name": "The Dark Knight"
})
print(response)


# ─────────────────────────────────────────────────────────────
# 3. FewShot PromptTemplate — Learn extraction pattern
# ─────────────────────────────────────────────────────────────
print("\n[3] FewShot PromptTemplate — Learn Extraction Pattern")
print("-" * 40)

examples = [
    {
        "movie": "Avengers: Endgame",
        "output": """
        Title        : Avengers: Endgame
        Director     : Anthony & Joe Russo
        Release Year : 2019
        Genre        : Action, Sci-Fi, Superhero
        IMDb Rating  : 8.4/10
        Box Office   : $2.798 billion
        Key Themes   : Sacrifice, Time, Loss, Redemption
        Iconic Line  : "I am Iron Man"
        """
    },
    {
        "movie": "The Godfather",
        "output": """
        Title        : The Godfather
        Director     : Francis Ford Coppola
        Release Year : 1972
        Genre        : Crime, Drama
        IMDb Rating  : 9.2/10
        Box Office   : $246 million
        Key Themes   : Power, Family, Loyalty, Corruption
        Iconic Line  : "I'm gonna make him an offer he can't refuse"
        """
    }
]

example_template = PromptTemplate(
    input_variables=["movie", "output"],
    template="Movie: {movie}\nExtracted Info:\n{output}"
)

fewshot_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="You are a movie information extractor. Learn the extraction pattern from these examples:",
    suffix="Movie: {movie_name}\nExtracted Info:",
    input_variables=["movie_name"]
)

prompt = fewshot_template.format(movie_name="Interstellar")
response = model.invoke(prompt)
print(response.content)


# ─────────────────────────────────────────────────────────────
# 4. FewShot ChatPromptTemplate — Q&A extraction style
# ─────────────────────────────────────────────────────────────
print("\n[4] FewShot ChatPromptTemplate — Q&A Extraction")
print("-" * 40)

chat_examples = [
    {
        "input":  "Tell me about Titanic",
        "output": "🎬 Titanic (1997) | Director: James Cameron | Genre: Romance/Drama | IMDb: 7.9 | Stars: Leonardo DiCaprio, Kate Winslet | Theme: Love vs survival | Won 11 Academy Awards"
    },
    {
        "input":  "Tell me about Parasite",
        "output": "🎬 Parasite (2019) | Director: Bong Joon-ho | Genre: Thriller/Drama | IMDb: 8.5 | Stars: Song Kang-ho | Theme: Class inequality | First non-English film to win Best Picture Oscar"
    },
    {
        "input":  "Tell me about Joker",
        "output": "🎬 Joker (2019) | Director: Todd Phillips | Genre: Crime/Drama | IMDb: 8.4 | Stars: Joaquin Phoenix | Theme: Mental illness, society failure | Won 2 Academy Awards"
    }
]

few_shot_chat = FewShotChatMessagePromptTemplate(
    example_prompt=ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai",    "{output}")
    ]),
    examples=chat_examples
)

final_template = ChatPromptTemplate.from_messages([
    ("system", "You are a compact movie information extractor. Always respond in a single structured line with key facts."),
    few_shot_chat,
    ("human", "{input}")
])

chain = final_template | model | parser
response = chain.invoke({"input": "Tell me about Oppenheimer"})
print(response)


# ─────────────────────────────────────────────────────────────
# 5. MessagesPlaceholder — Movie chatbot with memory
# ─────────────────────────────────────────────────────────────
print("\n[5] MessagesPlaceholder — Movie Chatbot with Memory")
print("-" * 40)

memory_template = ChatPromptTemplate.from_messages([
    ("system", """
        You are an expert movie information extractor and chatbot.
        You remember all movies discussed in this conversation.
        When asked about a movie, extract: cast, director, plot, themes, rating.
        When comparing movies, use previously discussed movies from chat history.
     """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    memory_template | model | parser,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

config = {"configurable": {"session_id": "movie_session_1"}}

# Multi-turn movie conversation
turns = [
    "Extract info about the movie Interstellar",
    "Who directed it?",
    "Compare it with Inception — which is better?"
]

for turn in turns:
    print(f"\n🧑 You   : {turn}")
    response = chain_with_memory.invoke({"input": turn}, config=config)
    print(f"🤖 Bot   : {response}")


# ─────────────────────────────────────────────────────────────
# 6. Dynamic Template — Extract by category
# ─────────────────────────────────────────────────────────────
print("\n[6] Dynamic Template — Extract by Category")
print("-" * 40)

EXTRACTION_MODES = {
    "basic":    "Extract title, director, year, genre, IMDb rating, and a 2-line plot summary.",
    "cast":     "Extract full cast list, character names, and each actor's role importance.",
    "themes":   "Extract deep themes, symbols, metaphors, and philosophical messages in the movie.",
    "awards":   "Extract all awards won, nominations, box office performance, and critical reception.",
    "compare":  "Compare the two movies in terms of plot, direction, acting, and cultural impact."
}

def extract_movie_info(mode: str, movie: str, movie2: str = None) -> str:
    instruction = EXTRACTION_MODES.get(mode, EXTRACTION_MODES["basic"])

    if mode == "compare" and movie2:
        user_msg = f"Compare '{movie}' and '{movie2}'"
    else:
        user_msg = f"Movie: {movie}"

    template = ChatPromptTemplate.from_messages([
        ("system", f"You are a movie information extractor. Task: {instruction}"),
        ("human",  "{user_msg}")
    ])

    chain = template | model | parser
    return chain.invoke({"user_msg": user_msg})


# Run all extraction modes
print("\n📋 BASIC EXTRACTION — Pulp Fiction")
print(extract_movie_info("basic", "Pulp Fiction"))

print("\n🎭 CAST EXTRACTION — The Avengers")
print(extract_movie_info("cast", "The Avengers"))

print("\n💡 THEMES EXTRACTION — The Matrix")
print(extract_movie_info("themes", "The Matrix"))

print("\n🏆 AWARDS EXTRACTION — Everything Everywhere All at Once")
print(extract_movie_info("awards", "Everything Everywhere All at Once"))

print("\n⚔️  COMPARE — Inception vs Interstellar")
print(extract_movie_info("compare", "Inception", "Interstellar"))


print("\n" + "=" * 60)
print("         🎬 END OF MOVIE INFORMATION EXTRACTOR")
print("=" * 60)