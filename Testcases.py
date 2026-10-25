from csat_pipeline import analyze_feedback

# Testing Analyzer function
result = analyze_feedback(
    "My flight was delayed and nobody informed us."
)

print(result)

# Testing Analyzer function
result1 = analyze_feedback(
    "There is a very good journey with the flight. Cast and crew are very respobsible"
)

print(result1)

# Testing Analyzer function
result2 = analyze_feedback(
    "People are not comfortable due to seat conjusted"
)

print(result2)