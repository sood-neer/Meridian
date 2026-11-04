#!/usr/bin/env python3
"""
Test script to have a conversation with the AI Academic Advisor.
Run with: python test_ai_conversation.py
Make sure the Flask LLM server is running on port 5002 first.
"""
import json
import requests
import sys

BASE_URL = "http://127.0.0.1:5002/llm"
USER_ID = "test-conversation-session"

# Diverse questions to test different capabilities
TEST_QUESTIONS = [
    "What are the prerequisites for CMPSC 131?",
    "What majors does Penn State Abington offer?",
    "When is the late drop deadline for Spring 2026?",
    "Who is the Penn State Abington chancellor?",
    "Recommend me courses for the next semester",
    "What are computer science classes at Penn State?",
    "Can you explain the entry to major process?",
    "What is the difference between CMPSC 121 and CMPSC 131?",
    "How many credits do I need to graduate?",
    "What GenEd categories does Penn State require?",
]


def send_message(question: str, conversation_history: list) -> tuple[str, list]:
    """Send a question to the AI and return (response, updated_history)."""
    try:
        resp = requests.post(
            f"{BASE_URL}/generate-response",
            json={
                "user_question": question,
                "conversation_history": conversation_history,
                "user_id": USER_ID,
            },
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("graduation_plan") or data.get("response") or "[No response]"
        history = data.get("conversation_history", conversation_history)
        return text, history
    except requests.exceptions.ConnectionError:
        return (
            "[ERROR] Could not connect. Is the Flask server running on port 5002?",
            conversation_history,
        )
    except requests.exceptions.Timeout:
        return "[ERROR] Request timed out.", conversation_history
    except Exception as e:
        return f"[ERROR] {e}", conversation_history


def main():
    print("=" * 60)
    print("AI Academic Advisor - Conversation Test")
    print("=" * 60)
    print("Ensure Flask is running: cd IAAS/LLMAcadamic-Advisor && python start_flask.py")
    print()

    history = []
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n--- Question {i}/{len(TEST_QUESTIONS)} ---")
        print(f"USER: {q}")
        response, history = send_message(q, history)
        preview = response[:400] + "..." if len(response) > 400 else response
        print(f"AI:   {preview}")
        if "[ERROR]" in response:
            print("\nStopping due to error. Start the server and try again.")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Conversation test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
