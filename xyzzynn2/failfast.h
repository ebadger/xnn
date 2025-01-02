#pragma once

#define DIE (*(UINT32 *)0 = __LINE__)
#define CheckConditionFailFast(__X)  if((__X) == FALSE)   { DIE; }
#define CheckOKFailFast(__X)         if((__X)!=0)         { DIE; }
