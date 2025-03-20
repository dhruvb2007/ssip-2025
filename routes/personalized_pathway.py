from flask import Blueprint, request, jsonify
import google.generativeai as genai
import numpy as np
import json
from Model.model_loader import load_model  # Import model loader

pathway_bp = Blueprint('personalized_pathway', __name__)

# Configure Gemini API Key
genai.configure(api_key="AIzaSyAmNRvyNqBdFtotLiD-i6rmVrXf1Gg9Rc4")

max_values = {
    "Gujarati_PAT": 20, "Gujarati_SAT": 80,
    "Mathematics_PAT": 20, "Mathematics_SAT": 80,
    "Science_PAT": 20, "Science_SAT": 80,
    "SocialScience_PAT": 20, "SocialScience_SAT": 80,
    "English_PAT": 20, "English_SAT": 80,
    "Hindi_PAT": 20, "Hindi_SAT": 80,
    "Attendance": 100, "Assessment": 100
}

def preprocess_input(input_data):
    return np.array([input_data[key] / max_values[key] for key in max_values]).reshape(1, -1)

def generate_learning_path(predicted_score, input_data):
    prompt = f"""
    A student has the following academic performance:
    {json.dumps(input_data, indent=2)}

    The predicted final performance score is {predicted_score}/100.

    Based on this, generate a **Personalized Learning Pathway** in JSON format. **Ensure the response strictly follows this structure:**

    ```json
    {{
        "Personalized_Pathway": {{
            "analysis": {{
                "overall_performance": "Overall performance analysis in 2-3 sentences.",
                "strengths": [
                    {{"subject": "Subject Name", "rationale": "Why it is a strength."}}
                ],
                "weaknesses": [
                    {{"subject": "Subject Name", "rationale": "Why it is a weakness."}}
                ]
            }},
            "personalized_learning_pathway": {{
                "additional_recommendations": ["List of improvement suggestions."],
                "resources": {{
                    "english": ["List of English study resources."],
                    "gujarati": ["List of Gujarati study resources."],
                    "hindi": ["List of Hindi study resources."],
                    "mathematics": ["List of Math study resources."],
                    "science": ["List of Science study resources."],
                    "social_science": ["List of Social Science study resources."],
                    "general": ["General learning techniques."]
                }},
                "study_plan": {{
                    "strategy_notes": "Overall study strategy explanation.",
                    "weekly_schedule": [
                        {{"day": "Monday", "activities": ["Activity 1", "Activity 2"]}}
                    ]
                }}
            }},
            "predicted_score": {predicted_score},
            "student_performance": {json.dumps(input_data, indent=2)}
        }},
        "Predicted_Final_Score": {round(predicted_score, 2)}
    }}
    ```

    **Return only valid JSON in this exact format, with no additional text.**
    """

    model_gemini = genai.GenerativeModel("gemini-2.0-flash")
    response = model_gemini.generate_content(prompt)

    if not response or not hasattr(response, "text"):
        return jsonify({"status": "error", "message": "No response from Gemini API"}), 500

    try:
        response_text = response.text.strip()
        response_text = response_text.replace("```json", "").replace("```", "").strip()  # Clean unwanted formatting
        json_response = json.loads(response_text)

        # Validate expected structure
        required_keys = {"Personalized_Pathway", "Predicted_Final_Score"}
        if not all(key in json_response for key in required_keys):
            raise ValueError("Invalid JSON structure from Gemini API")

        return jsonify(json_response)

    except (json.JSONDecodeError, ValueError) as e:
        # Fallback response in case of API failure
        fallback_response = {
            "Personalized_Pathway": {
                "analysis": {
                    "overall_performance": "Data not available due to API failure.",
                    "strengths": [{"subject": "Unknown", "rationale": "No data available."}],
                    "weaknesses": [{"subject": "Unknown", "rationale": "No data available."}]
                },
                "personalized_learning_pathway": {
                    "additional_recommendations": ["Improve study habits."],
                    "resources": {
                        "english": ["Basic grammar books"],
                        "gujarati": ["Gujarati grammar books"],
                        "hindi": ["Hindi grammar books"],
                        "mathematics": ["Basic math guides"],
                        "science": ["Introductory science materials"],
                        "social_science": ["History and civics textbooks"],
                        "general": ["Time management techniques"]
                    },
                    "study_plan": {
                        "strategy_notes": "Follow a structured study approach.",
                        "weekly_schedule": [{"day": "Monday", "activities": ["Self-study session"]}]
                    }
                },
                "predicted_score": predicted_score,
                "student_performance": input_data
            },
            "Predicted_Final_Score": round(predicted_score, 2)
        }

        return jsonify({
            "status": "error",
            "message": "Invalid JSON format from Gemini API",
            "details": str(e),
            "fallback_response": fallback_response
        }), 500

@pathway_bp.route('/personalized_pathway', methods=['GET'])
def personalized_pathway():
    try:
        model = load_model()  # Use cached model
        input_data = {key: float(request.args.get(key, 0)) for key in max_values}
        input_data_reshaped = preprocess_input(input_data)
        predicted_score = model.predict(input_data_reshaped)[0] * 100
        return generate_learning_path(predicted_score, input_data)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
