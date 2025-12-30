function Message({ message }) {
  return (
    <div
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-3xl ${
          message.role === 'user'
            ? 'bg-yellow-800 text-white'
            : 'bg-gray-300 text-black'
        }`}
      >
        {message.content}
      </div>
    </div>
  )
}

export default Message