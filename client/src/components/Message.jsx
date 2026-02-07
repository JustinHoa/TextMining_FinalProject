function Message({ message }) {
  // If it's a fact-check result (from assistant with structured data)
  if (message.role === 'assistant' && message.factCheckResult) {
    const { claim, status, explanation, confidence, evidence } = message.factCheckResult

    return (
      <div className="flex justify-start">
        <div className="max-w-3xl w-full bg-white rounded-2xl shadow-lg overflow-hidden border border-gray-200">
          {/* Header with Status Badge */}
          <div className="bg-gradient-to-r from-blue-50 to-yellow-50 p-6 border-b border-gray-200">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-3">
                <div className={`px-4 py-1.5 rounded-full font-semibold text-sm shadow-sm ${status === 'True'
                    ? 'bg-green-500 text-white'
                    : status === 'False'
                      ? 'bg-red-500 text-white'
                      : 'bg-yellow-500 text-white'
                  }`}>
                  {status}
                </div>
                <div className="flex items-center gap-2 text-gray-700">
                  <span className="text-sm font-medium">Độ tin cậy:</span>
                  <div className="flex items-center gap-1">
                    <div className="w-24 h-2 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${confidence >= 0.8 ? 'bg-green-500' : confidence >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
                          }`}
                        style={{ width: `${confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-semibold">{(confidence * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Claim */}
            <div className="bg-white rounded-lg p-4 border-l-4 border-yellow-500">
              <p className="text-sm text-gray-500 mb-1 font-medium">Phát biểu cần kiểm tra:</p>
              <p className="text-gray-800 font-medium leading-relaxed">{claim}</p>
            </div>
          </div>

          {/* Explanation */}
          <div className="p-6 bg-gray-50">
            <h3 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-2">
              <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Giải thích:
            </h3>
            <p className="text-gray-700 leading-relaxed">{explanation}</p>
          </div>

          {/* Evidence */}
          {evidence && evidence.length > 0 && (
            <div className="p-6 border-t border-gray-200">
              <h3 className="text-sm font-semibold text-gray-700 mb-4 flex items-center gap-2">
                <svg className="w-5 h-5 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Bằng chứng ({evidence.length}):
              </h3>
              <div className="space-y-3">
                {evidence.map((ev, idx) => (
                  <div key={idx} className="bg-white rounded-xl p-4 border border-gray-200 hover:border-yellow-300 transition-all hover:shadow-md">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2 flex-1">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${ev.trust_level === 'High'
                            ? 'bg-green-100 text-green-700'
                            : ev.trust_level === 'Medium'
                              ? 'bg-yellow-100 text-yellow-700'
                              : 'bg-gray-100 text-gray-700'
                          }`}>
                          {ev.trust_level || 'Unknown'}
                        </span>
                        <span className="text-xs text-gray-500">
                          {ev.source} • Điểm: {(ev.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    {ev.statement && (
                      <h4 className="font-medium text-gray-800 mb-2 text-sm">{ev.statement}</h4>
                    )}

                    {ev.evidence_chunk && (
                      <p className="text-sm text-gray-600 mb-2 line-clamp-3">{ev.evidence_chunk}</p>
                    )}

                    {ev.url && (
                      <a
                        href={ev.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs text-yellow-600 hover:text-yellow-800 hover:underline flex items-center gap-1"
                      >
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                        </svg>
                        Xem nguồn
                      </a>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    )
  }

  // Regular text message (user input or simple assistant message)
  return (
    <div
      className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div
        className={`max-w-xs lg:max-w-md px-4 py-2 rounded-3xl shadow-sm ${message.role === 'user'
            ? 'bg-gradient-to-r from-yellow-600 to-yellow-700 text-white'
            : 'bg-white text-gray-800 border border-gray-200'
          }`}
      >
        {message.content}
      </div>
    </div>
  )
}

export default Message