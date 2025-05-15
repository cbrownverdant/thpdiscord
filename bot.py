import os
import re
import openai
import discord
from discord.ext import commands
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# --------- API KEYS (Hardcoded for local testing) ---------
openai.api_key = os.getenv("OPENAI_API_KEY")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# --------- Config ---------
INDEX_FILE = "sop_index.faiss"
META_FILE = "sop_chunks.pkl"
MODEL_NAME = "gpt-3.5-turbo"  

# --------- Load FAISS Index & Metadata ---------
if not os.path.exists(INDEX_FILE) or not os.path.exists(META_FILE):
    raise FileNotFoundError("❌ Missing FAISS index or metadata file.")

index = faiss.read_index(INDEX_FILE)

with open(META_FILE, "rb") as f:
    metadata = pickle.load(f)

# --------- Embedding Model ---------
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# --------- Discord Setup ---------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ {bot.user.name} is online!")

# --------- Message Splitter ---------
def split_message(text, limit=2000):
    lines = text.split('\n')
    chunks = []
    current = ""
    for line in lines:
        if len(current) + len(line) + 1 <= limit:
            current += line + "\n"
        else:
            chunks.append(current.strip())
            current = line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks

# --------- Main Handler ---------
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if bot.user.mentioned_in(message) and "?" in message.content:
        user_question = message.content.replace(f"<@{bot.user.id}>", "").strip()

        try:
            # Step 1: Embed user question
            q_embedding = EMBEDDING_MODEL.encode([user_question])
            D, I = index.search(np.array(q_embedding).astype("float32"), k=8)

            # Step 2: Build clean context from top matches
            max_context_chars = 7000
            context_parts = []
            for i in I[0]:
                if 0 <= i < len(metadata):
                    excerpt = metadata[i]
                    cleaned = excerpt["content"]
                    if len(cleaned) > 50:
                        context_parts.append(f"- {cleaned.strip()}")

            if not context_parts:
                await message.channel.send(
                    f"📄 {message.author.mention} Sorry, I couldn’t find anything in the SOP to help with that. "
                    "Please contact your manager or refer to AGCO guidelines for clarification."
                )
                return

            # Truncate context
            context = "\n".join(context_parts)
            context = context[:max_context_chars]

            # Step 3: Construct OpenAI message
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for staff at The Hunny Pot Cannabis Co. "
            		"Only answer questions using the SOP excerpts provided below. "
            		"Provide only what is relevant to answer the user's question. Do not include unrelated or extra information. "
            		"If the SOP contains a step-by-step procedure, explain the process clearly in natural language. "
            		"When helpful for clarity, organize multi-step procedures into clear bullet points. "
			"If the SOP includes multiple procedures related to the same task (e.g., system steps in Cova and warehouse/packaging steps), label each set of steps accordingly so the user understands the different parts of the process. "
			"For example, you may begin sections with 'Steps in Cova' or 'Warehouse Preparation Steps' if that helps organize the information. "
            		"Do not copy the original SOP's formatting — summarize steps in your own words, clearly and practically. "
            		"If a link (URL) is present in the SOP excerpt and it's relevant to the question, always include it exactly as written. "
            		"Never invent or modify URLs. Only share links that appear in the SOP excerpts. "
            		"If the SOP does not contain an answer, say so directly and do not attempt to guess. "
            		"If names, phone numbers, or emails are shown in the SOP excerpts, include them only as written. "
            		"Do not mention SOP file names, document numbers, or section titles. "
            		"Never say things like 'according to the SOP' — just provide the correct procedure or information clearly and practically."
			"If a user asks about Thomas Kitchens, the weather, or what to do with their spare time, respond in a playful and lighthearted tone using the jokes and ideas provided in the SOP content."
                    )
                },
                {
                    "role": "user",
                    "content": f"Relevant SOP Excerpts:\n{context}\n\nUser Question: {user_question}"
                }
            ]

            # Step 4: Query OpenAI
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=600,
                temperature=0.5
            )

            answer = response.choices[0].message.content.strip()

            # Step 5: Send reply
            for chunk in split_message(f"🧠 {message.author.mention} {answer}"):
                await message.channel.send(chunk)

        except Exception as e:
            await message.channel.send(f"⚠️ Error: {str(e)}")
            print(f"Error: {str(e)}")

    await bot.process_commands(message)

# --------- Run Bot ---------
bot.run(DISCORD_TOKEN)

