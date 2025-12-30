import { useState, useRef, useEffect } from 'react'
import Message from './components/Message'
import ChatInput from './components/ChatInput'
import { getMockResponse } from './services/chatService'
import './index.css'
import NavBar from './components/NavBar'

function App() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! How can I help you today?' }
  ])
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const handleSendMessage = (content) => {
    const userMessage = { role: 'user', content }
    setMessages(prev => [...prev, userMessage])

    // Simulate AI response
    setTimeout(() => {
      const aiMessage = { role: 'assistant', content: getMockResponse() }
      setMessages(prev => [...prev, aiMessage])
    }, 1000)
  }

  return (
    <div className="p-4 items-center  bg-stone-300 gap-4  flex flex-col w-screen h-screen">
      {/* Nav Bar */}
      <NavBar />
      {/* Messages */}
      <div className="border border-stone-400 rounded-2xl bg-stone-200 max-w-[1260px] w-full flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className='max-w-[1260px] w-full'>

      <ChatInput onSendMessage={handleSendMessage} />
      </div>
    </div>
  )
}

export default App