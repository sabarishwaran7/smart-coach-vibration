#include <ESP8266WiFi.h>
#include <ThingSpeak.h>

const char* ssid = "vivo";      // ✅ Your phone hotspot name (exact spelling)
const char* password = "vivo1234";     // ✅ Your hotspot password

WiFiClient client;

unsigned long myChannelNumber = 3096181;            // ✅ Your ThingSpeak Channel Number
const char* myWriteAPIKey = "QLVJ86SMW0EL5NZE";     // ✅ Your Write API Key

void setup() {
  Serial.begin(115200);
  delay(100);

  WiFi.begin(ssid, password);
  Serial.println("Connecting to WiFi...");

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("✅ WiFi Connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());

  ThingSpeak.begin(client);
}

void loop() {
  // Sample dummy vibration values (later you can replace with sensor data)
  int ax = random(100, 200);
  int ay = random(200, 300);
  int az = random(300, 400);
  float temperature = random(25, 35);

  ThingSpeak.setField(1, ax);
  ThingSpeak.setField(2, ay);
  ThingSpeak.setField(3, az);
  ThingSpeak.setField(4, temperature);

  int statusCode = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);

  if (statusCode == 200) {
    Serial.println("✅ Data uploaded to ThingSpeak successfully!");
  } else {
    Serial.print("❌ Upload failed, HTTP error code: ");
    Serial.println(statusCode);
  }

  delay(15000);  // 15 sec delay (ThingSpeak limit)
}


