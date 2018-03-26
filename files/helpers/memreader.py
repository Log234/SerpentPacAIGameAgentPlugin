from ctypes import *
from ctypes.wintypes import *
from time import sleep
import win32api, win32ui, win32process
import ctypes
import ntpath

class MemoryReader:
    HealthAddress = -1
    ScoreAddress = -1
    PausedAddress = -1
    GameOverAddress = -1

    def __init__(self, **kwargs):
        OpenProcess = windll.kernel32.OpenProcess
        self.ReadProcessMemory = windll.kernel32.ReadProcessMemory
        SIZE_T = c_size_t
        self.ReadProcessMemory.argtypes = [HANDLE, LPCVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
        FindWindowA = windll.user32.FindWindowA
        GetWindowThreadProcessId = windll.user32.GetWindowThreadProcessId

        PROCESS_ALL_ACCESS = 0x1F0FFF
        HWND = win32ui.FindWindow(None,u"ARCADE GAME SERIES: PAC-MAN").GetSafeHwnd()
        PID = win32process.GetWindowThreadProcessId(HWND)[1]
        self.processHandle = ctypes.windll.kernel32.OpenProcess(PROCESS_ALL_ACCESS,False,PID)

        values = win32process.EnumProcessModules(self.processHandle)
        self.pointer_dict = dict()
        for value in values:
            name = ntpath.basename(win32process.GetModuleFileNameEx(self.processHandle, value))
            self.pointer_dict[name] = value

        PacManBaseAddress = self.pointer_dict["PAC-MAN.exe"]
        MonoBaseAddress = self.pointer_dict["mono.dll"]
        ReleaseBaseAddress = self.pointer_dict["Release_3.dll"]
        TierBaseAddress = self.pointer_dict["tier0_s64.dll"]

        #print(f"HWND: {HWND}")
        #print(f"PID: {PID}")
        #print(f"PROCESS: {self.processHandle}")
        #print(f"PacManBaseAddress: {PacManBaseAddress}")
        #print(f"MonoBaseAddress: {MonoBaseAddress}")
        #print(f"ReleaseBaseAddress: {ReleaseBaseAddress}")

        tmp = c_int()

        healthPath = [0x3C440, 0x170, 0x8, 0x20, 0x0, 0x18]
        print("Getting address of Lives")
        self.HealthAddress = self.GetAddress(ReleaseBaseAddress, healthPath)

        scorePath = [0x169FB8, 0x110, 0x0, 0x258]
        print("Getting address of Score")
        self.ScoreAddress = self.GetAddress(TierBaseAddress, scorePath)

        pausedPath = [0x1125428, 0x8, 0x328, 0x130]
        print("Getting address of Paused")
        self.PausedAddress = self.GetAddress(PacManBaseAddress, pausedPath)

        gameOverPath = [0x260110, 0x178, 0x5C]
        print("Getting address of Game Over")
        self.GameOverAddress = self.GetAddress(MonoBaseAddress, gameOverPath)

        #print(self.HealthAddress)
        #print(self.ScoreAddress)
        #print(self.PausedAddress)
        #print(self.GameOverAddress)


    def GetAddress(self, basePointer, offsets):
        numRead = c_size_t()
        tmp = c_int()
        print(f"BasePointer: {basePointer}")
        if not self.ReadProcessMemory(self.processHandle, basePointer + offsets[0], byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read a location in the memory. Address: {basePointer + offsets[0]}")
        print(f"Tmp: {tmp.value}")
        for i in range(1, offsets.__len__() - 1):
            if not self.ReadProcessMemory(self.processHandle, tmp.value + offsets[i], byref(tmp), 4, byref(numRead)):
                raise RuntimeError(f"Failed to read a location in the memory. Address: {tmp.value + offsets[i]}")
            print(f"Tmp: {tmp.value}")

        return tmp.value + offsets[offsets.__len__()-1]

    def GetLives(self):
        numRead = c_size_t()
        tmp = c_int()
        if not self.ReadProcessMemory(self.processHandle, self.HealthAddress, byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read the health from memory. Address: {self.HealthAddress}")

        return tmp.value

    def GetScore(self):
        numRead = c_size_t()
        tmp = c_int()
        if not self.ReadProcessMemory(self.processHandle, self.ScoreAddress, byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read the score from memory. Address: {self.ScoreAddress}")

        return tmp.value

    def GetPaused(self):
        numRead = c_size_t()
        tmp = c_int()
        if not self.ReadProcessMemory(self.processHandle, self.PausedAddress, byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read pause status from memory. Address: {self.PausedAddress}")

        return tmp.value
        
    def GetGameOver(self):
        numRead = c_size_t()
        tmp = c_int()
        if not self.ReadProcessMemory(self.processHandle, self.GameOverAddress, byref(tmp), 4, byref(numRead)):
            raise RuntimeError(f"Failed to read the game status from memory. Address: {self.GameOverAddress}")

        return tmp.value