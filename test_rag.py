from src.chat import ask_question


# ==================================================
# TEST CASES
# ==================================================

TEST_CASES = [

    {
        "question": "Haryana mein kitne dealers hain?",
        "expected": "8"
    },

    {
        "question": "Aditya Enterprises ka mobile number kya hai?",
        "expected": "9419283110"
    },

    {
        "question": "Aditya Enterprises ka GST kya hai?",
        "expected": "01AQNPS8286M1Z0"
    },

    {
        "question": "Aditya Enterprises ka email kya hai?",
        "expected": "adityabuildmart@gmail.com"
    },

    {
        "question": "MBC37SBC ki maximum power kitni hai?",
        "expected": "1.35 HP"
    },

    {
        "question": "MBC37SBC ka displacement kitna hai?",
        "expected": "37.7 cc"
    },

    {
        "question": "MCS58A-22SN ki maximum power kitni hai?",
        "expected": "3.2 HP"
    },

    {
        "question": "Bonhoeffer ke catalogue mein kaun-kaun se products hain?",
        "expected": "Multi-Tool"
    },

    {
        "question": "Bonhoeffer ka email address kya hai?",
        "expected": "sales@bonhoeffermachines.com"
    }
]


# ==================================================
# RUN TESTS
# ==================================================

print("================================")
print("RAG AUTOMATIC TEST")
print("================================")


passed = 0
failed = 0


for index, test in enumerate(
    TEST_CASES,
    start=1
):

    question = test["question"]

    expected = test["expected"]


    print()
    print(
        f"Test {index}/{len(TEST_CASES)}"
    )

    print(
        "Question:",
        question
    )


    try:

        # Fresh history for every test
        history = []


        answer = ask_question(
            question,
            history
        )


        print(
            "Answer:",
            answer
        )


        # Case-insensitive comparison
        answer_lower = (
            str(answer)
            .lower()
        )


        expected_lower = (
            str(expected)
            .lower()
        )


        if expected_lower in answer_lower:

            print(
                "RESULT: PASS"
            )

            passed += 1

        else:

            print(
                "RESULT: FAIL"
            )

            print(
                "Expected:",
                expected
            )

            failed += 1


    except Exception as e:

        print(
            "RESULT: ERROR"
        )

        print(
            "Error:",
            e
        )

        failed += 1


# ==================================================
# SUMMARY
# ==================================================

print()
print("================================")
print("TEST SUMMARY")
print("================================")

print(
    "Total:",
    len(TEST_CASES)
)

print(
    "Passed:",
    passed
)

print(
    "Failed:",
    failed
)

print("================================")


if failed == 0:

    print(
        "ALL TESTS PASSED"
    )

else:

    print(
        "SOME TESTS FAILED"
    )