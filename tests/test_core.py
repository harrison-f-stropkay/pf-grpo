from pf_grpo.core import Task, _message, _task, make_messages, reward_math


def test_make_messages_includes_problem_and_boxed_instruction() -> None:
    messages = make_messages(Task("What is 2+2?", "4"))

    assert messages == [
        {
            "role": "user",
            "content": "Solve the following math problem. Show your work, and put the final answer in \\boxed{}.\n\nWhat is 2+2?",
        }
    ]


def test_reward_math_verifies_boxed_answer() -> None:
    task = Task("What is 2+2?", "4")

    assert reward_math("The answer is \\boxed{4}.", task) == 1.0
    assert reward_math("The answer is \boxed{4}.", task) == 0.0
    assert reward_math("The answer is \\boxed{4.0}.", task) == 1.0
    assert reward_math("The answer is \\boxed{5}.", task) == 0.0


def test_task_parses_falsey_zero_answer_without_falling_through() -> None:
    assert _task({"problem": "What is 0?", "answer": 0}) == Task("What is 0?", "0")
    assert _task({"problem": "What is false?", "answer": False}) == Task("What is false?", "False")
    assert _task({"problem": "What is empty?", "answer": ""}) == Task("What is empty?", "")


class ChatMessageLike:
    role = "assistant"

    def extract_text_content(self) -> str:
        return "hello from an object"


def test_message_normalizes_dicts_and_chat_message_like_objects() -> None:
    assert _message({"role": "user", "content": "hello from a dict"}) == {
        "role": "user",
        "content": "hello from a dict",
    }
    assert _message(ChatMessageLike()) == {
        "role": "assistant",
        "content": "hello from an object",
    }
