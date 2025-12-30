const mockResponses = [
  "That's an interesting question! Let me think about that.",
  "I understand. Can you tell me more about what you're looking for?",
  "Great point! Here's what I think...",
  "Thanks for sharing that. Based on what you've said, I suggest...",
  "I'm here to help. What else would you like to know?"
]

export function getMockResponse() {
  return mockResponses[Math.floor(Math.random() * mockResponses.length)]
}