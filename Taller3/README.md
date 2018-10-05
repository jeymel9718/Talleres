# Taller 3
## NEON + OpenMP

Para este taller es necesario del uso del Android NDK y el Android Debug Bridge (ADB)

## Ejecución
Para la ejecución es necesario de un dispositivo Android, la conexión se realiza por medio de USB y el dispositivo
debe estar en modo de depuración

## Comandos
/opt/android-ndk-r16b/ndk-build

adb push ../libs/armeabi-v7a/pi_par /data/local/tmp

adb shell /data/local/tmp/pi_par
