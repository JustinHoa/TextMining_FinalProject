const API_BASE_URL = 'http://localhost:8000'

/**
 * Verify a claim using the ViFactCheck API
 * @param {string} claim - The claim to verify
 * @returns {Promise<Object>} The verification result
 */
export async function verifyClaimAPI(claim) {
  try {
    const response = await fetch(`${API_BASE_URL}/check`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ claim }),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}))
      throw new Error(errorData.detail || `HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    return data
  } catch (error) {
    console.error('Error verifying claim:', error)
    throw error
  }
}

/**
 * Check API health
 * @returns {Promise<Object>} Health check response
 */
export async function checkAPIHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/`)
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }
    return await response.json()
  } catch (error) {
    console.error('API health check failed:', error)
    throw error
  }
}