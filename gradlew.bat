@ECHO OFF
SETLOCAL
SET DIR=%~dp0
SET APP_BASE_NAME=%~n0
SET APP_HOME=%DIR%

IF NOT EXIST "%APP_HOME%\gradle\wrapper\gradle-wrapper.jar" (
  ECHO Gradle wrapper jar not found.
  EXIT /B 1
)

"%JAVA_HOME%\bin\java.exe" -Xmx64m -jar "%APP_HOME%\gradle\wrapper\gradle-wrapper.jar" %*
