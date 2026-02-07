import { useState } from 'react'

function ChatInput({ onSendMessage, disabled = false }) {
  const [input, setInput] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || disabled) return
    onSendMessage(input)
    setInput('')
  }

  return (
    <div className="border border-gray-300 rounded-2xl p-2 bg-white shadow-lg">
      <form onSubmit={handleSubmit} className="flex space-x-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Nhập tuyên bố hoặc tin tức cần kiểm tra..."
          disabled={disabled}
          className="bg-gray-50 text-gray-800 flex-1 px-4 py-3 border border-gray-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-yellow-800 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed transition-all"
        />
        <button
          type="submit"
          disabled={disabled}
          className="px-6 py-3 bg-yellow-600 text-white rounded-2xl hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-yellow-800 focus:ring-offset-2 active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100 transition-all font-medium shadow-md"
        >
          Kiểm tra
        </button>
      </form>
    </div>
  )
}

export default ChatInput