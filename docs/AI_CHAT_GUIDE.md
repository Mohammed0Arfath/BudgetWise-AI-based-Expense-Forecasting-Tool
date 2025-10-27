# 🤖 AI Budget Advisor - Setup Guide
**© 2025 Mohammed Arfath - BudgetWise AI**

## Overview
The AI Budget Advisor integrates Google Gemini AI to provide personalized budget recommendations, spending insights, and interactive financial advice through a chat interface.

## Features

### 💬 Interactive Chat
- Natural language conversations about your finances
- Personalized budget recommendations
- Spending pattern analysis
- Money-saving tips and strategies
- Goal-setting assistance

### ⚡ Quick Insights
- AI-generated spending pattern analysis
- Automatic saving opportunity detection
- Personalized budget recommendations

### 💰 Savings Plan Creator
- Custom savings plans based on your goals
- Month-by-month milestones
- Feasibility analysis
- Actionable strategies to achieve targets

## Setup Instructions

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install google-generativeai streamlit python-dotenv
```

### 2. Get Your Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

### 3. Configure API Key

**Option A: Environment Variable (Recommended)**

Create a `.env` file in the project root:
```bash
cp .env.example .env
```

Edit `.env` and add your API key:
```
GEMINI_API_KEY=your_actual_api_key_here
```

**Option B: Direct Input**

When running the app, enter your API key in the sidebar under "🔑 API Configuration"

### 4. Run the Application

```bash
streamlit run app/budgetwise_app.py
```

Navigate to "🤖 AI Chat" in the sidebar.

## Usage Examples

### Example Conversations

**Budget Analysis:**
```
You: "Can you analyze my spending patterns?"
AI: "Based on your data, I notice you spend an average of ₹2,450 daily, 
     with weekends showing 35% higher spending. Your top categories are..."
```

**Saving Tips:**
```
You: "How can I save ₹20,000 in 3 months?"
AI: "To save ₹20,000 in 3 months, you'll need to save ₹6,667/month. 
     Here's a practical plan..."
```

**Budget Recommendations:**
```
You: "What should my daily budget be?"
AI: "Based on your spending patterns, I recommend a daily budget of ₹2,200,
     which is 10% lower than your current average..."
```

### Quick Insights

Click "🔄 Generate Fresh Insights" to get:
- **Spending Pattern**: Key trends and observations
- **Saving Opportunity**: Specific ways to reduce expenses
- **Budget Recommendation**: Realistic daily/monthly budget limits

### Savings Plan

1. Enter your savings goal (₹)
2. Select time period (months)
3. Click "🚀 Create Savings Plan"
4. Get personalized strategies and milestones

## Features & Capabilities

### What the AI Can Do:
✅ Analyze spending patterns and trends
✅ Identify saving opportunities
✅ Suggest realistic budgets
✅ Create customized savings plans
✅ Answer questions about your finances
✅ Provide category-specific recommendations
✅ Help set financial goals
✅ Offer practical money management tips

### What the AI Cannot Do:
❌ Access your bank accounts directly
❌ Make financial transactions
❌ Provide investment advice (not a licensed advisor)
❌ Guarantee specific financial outcomes
❌ Access data outside your uploaded expenses

## Privacy & Security

- ✅ Your expense data is processed locally
- ✅ Only anonymized summaries are sent to Gemini AI
- ✅ No personal identifying information is shared
- ✅ API keys are stored securely in environment variables
- ✅ Chat history is session-based (cleared on refresh)

## Troubleshooting

### "Gemini AI is not available"
**Solution:** Install the package
```bash
pip install google-generativeai
```

### "Please enter your Gemini API key"
**Solution:** 
1. Get API key from https://makersuite.google.com/app/apikey
2. Add to `.env` file or enter in sidebar

### "Failed to initialize AI advisor"
**Possible causes:**
- Invalid API key
- Network connection issues
- API quota exceeded (check Google AI Studio)

**Solution:** Verify API key and check network connection

### "Rate limit exceeded"
**Solution:** 
- Wait a few minutes
- Check your API quota in Google AI Studio
- Consider upgrading to paid tier for higher limits

## API Costs

**Gemini 1.5 Flash** (Currently used):
- **Free tier**: 15 requests per minute
- **Cost**: Free up to generous limits
- **Perfect for**: Personal budget tracking

For detailed pricing: https://ai.google.dev/pricing

## Best Practices

1. **Be Specific**: Ask clear, specific questions for better advice
2. **Provide Context**: Mention your goals when asking for recommendations
3. **Regular Updates**: Generate fresh insights periodically
4. **Set Realistic Goals**: Use the savings planner for achievable targets
5. **Review Suggestions**: AI advice is a tool, use your judgment

## Example Use Cases

### 1. Monthly Budget Review
```
"Review my spending for the last month and suggest improvements"
```

### 2. Category Optimization
```
"I'm spending too much on food. What can I do?"
```

### 3. Emergency Fund Planning
```
"Help me build a 3-month emergency fund"
```

### 4. Expense Reduction
```
"Where can I cut expenses without affecting my lifestyle?"
```

## Support

For issues or questions:
- GitHub: https://github.com/Mohammed0Arfath/BudgetWise-AI-based-Expense-Forecasting-Tool
- Check existing issues or create a new one

## License

This feature is part of BudgetWise AI
© 2025 Mohammed Arfath
MIT License with Attribution Requirement

---

**Built with:**
- 🤖 Google Gemini AI
- 🎨 Streamlit
- 🐍 Python

**Happy Budgeting! 💰✨**
