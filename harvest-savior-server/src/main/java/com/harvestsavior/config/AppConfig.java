package com.harvestsavior.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;

/**
 * AppConfig — registers Spring Beans that are not components themselves.
 *
 * @Configuration marks this class as a configuration source, similar to a
 * Spring XML application context file, but written in Java.
 *
 * @Bean tells Spring to manage the returned object's lifecycle.
 * Any class that needs a RestTemplate can declare it as a constructor parameter
 * and Spring will automatically inject this instance — this is called
 * Dependency Injection (DI), a core Spring principle.
 *
 * WHY a Bean instead of "new RestTemplate()" everywhere?
 * → A single shared instance is more efficient (no repeated object creation),
 *   and it can be configured globally (timeouts, interceptors) in one place.
 */
@Configuration
public class AppConfig {

    @Bean
    public RestTemplate restTemplate() {
        return new RestTemplate();
    }
}
