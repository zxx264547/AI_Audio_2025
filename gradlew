#!/usr/bin/env sh

set -e

APP_HOME=$(cd "$(dirname "$0")" && pwd)
GRADLE_WRAPPER_JAR="$APP_HOME/gradle/wrapper/gradle-wrapper.jar"
JAVA_OPTS=${JAVA_OPTS:-"-Xmx64m"}

if [ ! -f "$GRADLE_WRAPPER_JAR" ]; then
  echo "Gradle wrapper jar not found: $GRADLE_WRAPPER_JAR" >&2
  exit 1
fi

exec "$JAVA_HOME/bin/java" $JAVA_OPTS -jar "$GRADLE_WRAPPER_JAR" "$@"
