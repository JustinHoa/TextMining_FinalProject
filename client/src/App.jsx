import { useState, useRef, useEffect } from 'react'
import Message from './components/Message'
import ChatInput from './components/ChatInput'
import { verifyClaimAPI } from './services/chatService'
import './index.css'
import NavBar from './components/NavBar'

function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Xin chào! Tôi là Factifier - Hệ thống kiểm tra thông tin. Hãy nhập tuyên bố hoặc tin tức bạn muốn kiểm tra.'
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(scrollToBottom, [messages])

  const handleSendMessage = async (content) => {
    // Add user message
    const userMessage = { role: 'user', content }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Call the fact-checking API
      const result = await verifyClaimAPI(content)

      // Add AI response with fact-check result
      const aiMessage = {
        role: 'assistant',
        content: '', // Empty content since we're using factCheckResult
        factCheckResult: result
      }
      setMessages(prev => [...prev, aiMessage])
    } catch (error) {
      // Add error message
      const errorMessage = {
        role: 'assistant',
        content: `⚠️ Đã xảy ra lỗi khi kiểm tra thông tin: ${error.message}. Vui lòng thử lại hoặc kiểm tra xem API có đang chạy không.`
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div
      className="p-4 items-center bg-gradient-to-br from-gray-50 to-gray-100 gap-4 flex flex-col w-screen h-screen"
      style={{ 
        backgroundImage: 'url(/bg-tet.jpg)',
        backgroundSize: 'cover',
        backgroundPosition: 'top',
        backgroundRepeat: 'no-repeat'
      }}
    > 
      {/* Nav Bar */}
      <NavBar />

      {/* Messages */}
      <div className="border border-gray-300 rounded-2xl bg-white max-w-[1260px] w-full flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar shadow-xl">
        {messages.map((message, index) => (
          <Message key={index} message={message} />
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-gray-200 rounded-2xl px-6 py-4 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-yellow-600 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-yellow-600 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-yellow-600 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
                <span className="text-gray-600 text-sm">Đang kiểm tra thông tin...</span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className='max-w-[1260px] w-full'>
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      </div>
    </div>
  )
}

export default App