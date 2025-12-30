import { useState } from 'react'

function ChatInput({ onSendMessage }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim()) return
    onSendMessage(input)
    setInput('')
  }

  return (
    <div className="border border-stone-400 rounded-2xl p-2">
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          className="bg-stone-200 text-black flex-1 px-4 py-2 border border-gray-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-amber-600"
        />
        <button
          type="submit"
          className=" px-6 py-2 bg-yellow-600 text-white rounded-2xl hover:bg-yellow-500 focus:outline-none focus:outline-2 focus:outline-amber-600 active:scale-95"
        >
          Send
        </button>
      </form>
    </div>
  )
}

export default ChatInput