function NavBar() {
  return (
      <nav className="bg-yellow-700 rounded-2xl p-4 flex items-center justify-between max-w-[1260px] w-full">
        <div className="flex items-center space-x-4">
          {/* Logo */}
          <div className="w-10 h-10 border-white border-2 rounded-full flex items-center justify-center">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          {/* App Name */}
          <h1 className="text-2xl font-bold text-gray-200">Fact-Checker</h1>
        </div>
      </nav>
  );
}
export default NavBar;