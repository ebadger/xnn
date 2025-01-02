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
	UINT32 uiCount = 10;
	UINT32 initSize = 0;
	m_uiHeaderBytes = uiHeaderBytes;

	if (!wzFileName)
	{
		return E_INVALIDARG;
	}

	StringCchPrintf(m_wcPath, _countof(m_wcPath), wzFileName);

	m_hFile = CreateFile(m_wcPath,
						GENERIC_WRITE | GENERIC_READ,
						FILE_SHARE_READ,
						nullptr,
						OPEN_EXISTING, 
						FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, 
						NULL);

	if (INVALID_HANDLE_VALUE == m_hFile || 0 == m_hFile)
	{
		return E_FAIL;
	}

	CheckConditionFailFast(m_hFile != nullptr && m_hFile != INVALID_HANDLE_VALUE);
	
	m_hMap = CreateFileMapping(
		m_hFile,
		nullptr,                 // default security
		PAGE_READWRITE,          // read/write access
		0,                       // max. object size
		initSize,                // buffer size
		m_wcPath);               // name of mapping object

	CheckConditionFailFast(m_hMap != NULL && m_hMap != INVALID_HANDLE_VALUE);

	m_pBuf = (BYTE *)MapViewOfFile(m_hMap,   // handle to map object
		FILE_MAP_ALL_ACCESS, // read/write permission
		0,
		0,
		0);

	CheckConditionFailFast(m_pBuf != nullptr);

	UINT32 uiTemp = 0;

	m_uiMagicNumber = Utils::LittleToBigEndian(*(UINT32 *)m_pBuf);

	m_uiCount = Utils::LittleToBigEndian(*(UINT32 *)(m_pBuf + sizeof(UINT32)));
	m_pItems = (T *)(m_pBuf + m_uiHeaderBytes);

	return S_OK;
}

template <class T>
HRESULT FileMappingReadOnly<T>::Cleanup()
{
	CloseHandle(m_hMap);
	CloseHandle(m_hFile);

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
