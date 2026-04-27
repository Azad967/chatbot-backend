import os
from supabase import create_client
from dotenv import load_dotenv

# 🔹 Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# 🔹 Create client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)