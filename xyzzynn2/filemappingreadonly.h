#pragma once

template <class T>
class FileMappingReadOnly
{
public:
	HRESULT Initialize(const WCHAR *wzFileName, UINT32 uiHeaderBytes);
	HRESULT Cleanup();
	T * GetItem(UINT32 uiIndex);
	UINT32 Items();

private:
	WCHAR     m_wcPath[MAX_PATH];
	UINT32    m_uiMagicNumber;
	UINT32    m_uiHeaderBytes;
	UINT32    m_uiCount;
	HANDLE    m_hFile;
	HANDLE    m_hMap;
	BYTE   *  m_pBuf;
	T      *  m_pItems;
};

template <class T>
HRESULT FileMappingReadOnly<T>::Initialize(const WCHAR *wzFileName, UINT32 uiHeaderBytes)
{
	m_uiHeaderBytes = uiHeaderBytes;
	m_uiCount       = 0;
	m_pItems        = nullptr;
	m_pBuf          = nullptr;
	m_hMap          = nullptr;
	m_hFile         = INVALID_HANDLE_VALUE;

	if (!wzFileName)
	{
		return E_INVALIDARG;
	}

	StringCchPrintf(m_wcPath, _countof(m_wcPath), wzFileName);

	// Open truly read-only with full sharing so multiple instances of the
	// process can map the same MNIST data files concurrently.
	m_hFile = CreateFile(m_wcPath,
		GENERIC_READ,
		FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
		nullptr,
		OPEN_EXISTING,
		FILE_ATTRIBUTE_NORMAL,
		NULL);

	if (INVALID_HANDLE_VALUE == m_hFile)
	{
		DWORD err = GetLastError();
		WCHAR cdir[MAX_PATH] = {0};
		GetCurrentDirectoryW(MAX_PATH, cdir);
		wprintf(L"FileMappingReadOnly: CreateFile failed for %s in %s (error %lu%s)\r\n",
			m_wcPath, cdir, err,
			err == ERROR_SHARING_VIOLATION ? L" - file locked by another process" :
			err == ERROR_FILE_NOT_FOUND    ? L" - file not found" :
			err == ERROR_ACCESS_DENIED     ? L" - access denied" : L"");
		m_hFile = INVALID_HANDLE_VALUE;
		return HRESULT_FROM_WIN32(err);
	}

	// Unnamed read-only mapping. Naming the mapping with the file path was
	// both invalid (paths contain ':' / '\\') and a guaranteed collision
	// across processes; an anonymous mapping is the right thing here.
	m_hMap = CreateFileMapping(
		m_hFile,
		nullptr,
		PAGE_READONLY,
		0,
		0,
		nullptr);

	if (m_hMap == NULL)
	{
		DWORD err = GetLastError();
		wprintf(L"FileMappingReadOnly: CreateFileMapping failed for %s (error %lu)\r\n",
			m_wcPath, err);
		CloseHandle(m_hFile);
		m_hFile = INVALID_HANDLE_VALUE;
		return HRESULT_FROM_WIN32(err);
	}

	m_pBuf = (BYTE *)MapViewOfFile(m_hMap,
		FILE_MAP_READ,
		0,
		0,
		0);

	if (m_pBuf == nullptr)
	{
		DWORD err = GetLastError();
		wprintf(L"FileMappingReadOnly: MapViewOfFile failed for %s (error %lu)\r\n",
			m_wcPath, err);
		CloseHandle(m_hMap);
		CloseHandle(m_hFile);
		m_hMap  = nullptr;
		m_hFile = INVALID_HANDLE_VALUE;
		return HRESULT_FROM_WIN32(err);
	}

	m_uiMagicNumber = Utils::LittleToBigEndian(*(UINT32 *)m_pBuf);
	m_uiCount       = Utils::LittleToBigEndian(*(UINT32 *)(m_pBuf + sizeof(UINT32)));
	m_pItems        = (T *)(m_pBuf + m_uiHeaderBytes);

	return S_OK;
}

template <class T>
HRESULT FileMappingReadOnly<T>::Cleanup()
{
	if (m_pBuf)
	{
		UnmapViewOfFile(m_pBuf);
		m_pBuf = nullptr;
	}
	if (m_hMap)
	{
		CloseHandle(m_hMap);
		m_hMap = nullptr;
	}
	if (m_hFile != INVALID_HANDLE_VALUE && m_hFile != nullptr)
	{
		CloseHandle(m_hFile);
		m_hFile = INVALID_HANDLE_VALUE;
	}

	return S_OK;
}

template <class T>
T * FileMappingReadOnly<T>::GetItem(UINT32 uiItem)
{
	if (uiItem > m_uiCount)
	{
		return nullptr;
	}

	return &m_pItems[uiItem];
}

template <class T>
UINT32 FileMappingReadOnly<T>::Items()
{
	return m_uiCount;
}
