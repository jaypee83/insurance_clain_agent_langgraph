from langchain_core.prompts import ChatPromptTemplate

DOC_REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "TASK: DOC_REVIEW\nYou are an insurance document examiner. Compare the claim narrative with the extracted document fields. "
     "Report only concrete inconsistencies (names, dates, amounts, event type). Do not invent problems.\n{format_instructions}"),
    ("human", "CLAIM:\n{claim}\n\nDOCUMENTS:\n{documents}"),
])

COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "TASK: COVERAGE\nYou are a policy wording specialist. Decide whether the incident is a covered peril and whether any listed "
     "exclusion applies. Use ONLY the policy wording supplied. Be conservative: if unsure, lower your confidence.\n{format_instructions}"),
    ("human", "POLICY TYPE: {policy_type}\nCOVERED PERILS: {perils}\nEXCLUSIONS: {exclusions}\n\n"
     "INCIDENT TYPE: {incident_type}\nDESCRIPTION: {description}"),
])

FRAUD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "TASK: FRAUD_NARRATIVE\nYou are a Special Investigations Unit analyst. Assess the claim narrative for fraud red flags "
     "(e.g., cash payments, lost originals, related-party providers, vague details, pressure for fast payment). "
     "Rule-based indicators have already been computed; do not repeat them, add only narrative insights.\n{format_instructions}"),
    ("human", "CLAIM:\n{claim}\n\nRULE-BASED INDICATORS ALREADY FOUND:\n{rule_flags}"),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "TASK: SUMMARY\nYou are a senior claims adjuster. Write a concise, factual claim summary for the file, based ONLY on the "
     "check results provided. The recommended_action MUST equal the ROUTE given.\n{format_instructions}"),
    ("human", "ROUTE: {route}\nCLAIM:\n{claim}\n\nDOCUMENT CHECK:\n{doc_check}\n\nELIGIBILITY:\n{eligibility}\n\n"
     "FRAUD:\n{fraud}\n\nDECISION REASONS:\n{reasons}"),
])
