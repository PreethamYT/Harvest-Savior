package com.harvestsavior;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Entry point of the Spring Boot application.
 *
 * @SpringBootApplication is a convenience annotation that combines:
 *   1. @Configuration   — marks this class as a source of Spring Beans
 *   2. @EnableAutoConfiguration — tells Spring Boot to auto-configure based on classpath
 *   3. @ComponentScan   — scans this package and sub-packages for Spring components
 *
 * When the JVM starts, Spring Boot embeds a Tomcat server (port 8080 by default),
 * registers all controllers, services, and repositories, and starts listening for HTTP requests.
 */
@SpringBootApplication
public class HarvestSaviorApplication {

    public static void main(String[] args) {
        SpringApplication.run(HarvestSaviorApplication.class, args);
    }
}
