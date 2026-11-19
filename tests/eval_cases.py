CASES = [
    {
        "id": "meal_limit_current",
        "question": "What is the current travel meal limit per day?",
        "expected_doc": "expense_reimbursement_policy.md",
        "expected_chunk_ids": [
            "expense_reimbursement_policy.md::structural::4",
        ],
        "expected_answer_keys": ["50"],
    },
    {
        "id": "meal_code",
        "question": "EX-MEAL-50",
        "expected_doc": "expense_reimbursement_policy.md",
        "expected_chunk_ids": [
            "expense_reimbursement_policy.md::structural::5",
        ],
        "expected_answer_keys": ["EX-MEAL-50"],
    },
    {
        "id": "pto_accrual",
        "question": "How much PTO do full-time employees accrue per month in the first years?",
        "expected_doc": "pto_policy.md",
        "expected_chunk_ids": [
            "pto_policy.md::structural::3",
        ],
        "expected_answer_keys": ["1.5"],
    },
    {
        "id": "pto_carryover",
        "question": "How many PTO days can carry over to the next year?",
        "expected_doc": "pto_policy.md",
        "expected_chunk_ids": [
            "pto_policy.md::structural::5",
        ],
        "expected_answer_keys": ["5"],
    },
    {
        "id": "remote_stipend",
        "question": "What is the monthly home-office stipend for fully remote employees?",
        "expected_doc": "remote_work_policy.md",
        "expected_chunk_ids": [
            "remote_work_policy.md::structural::4",
        ],
        "expected_answer_keys": ["75"],
    },
    {
        "id": "remote_eligibility",
        "question": "How many months before an employee can request remote work?",
        "expected_doc": "remote_work_policy.md",
        "expected_chunk_ids": [
            "remote_work_policy.md::structural::2",
        ],
        "expected_answer_keys": ["six|6"],
    },
    {
        "id": "expense_deadline",
        "question": "Within how many days must expense reports be submitted under the current policy?",
        "expected_doc": "expense_reimbursement_policy.md",
        "expected_chunk_ids": [
            "expense_reimbursement_policy.md::structural::6",
        ],
        "expected_answer_keys": ["30"],
    },
    {
        "id": "lodging_cap",
        "question": "When does lodging need director pre-approval under the current expense policy?",
        "expected_doc": "expense_reimbursement_policy.md",
        "expected_chunk_ids": [
            "expense_reimbursement_policy.md::structural::4",
        ],
        "expected_answer_keys": ["300"],
    },
]