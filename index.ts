/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { gemini15Flash, googleAI, gemini20Flash} from '@genkit-ai/googleai';
import { genkit, z } from 'genkit';
import { startFlowServer } from '@genkit-ai/express';

const ai = genkit({
  plugins: [googleAI()],
});

export const MenuSuggestionFlow = ai.defineFlow(
  {
    name : "MenuSuggestionFlow"
  },
  async (restaurantTheme) => {
    const {text} = await ai.generate({
      model: gemini15Flash,
      // prompt: `Invent a menu item for a ${restaurantTheme} themed restaurant.`,

      // THESE TWO PROMPTS GIVE DECENT GENERALISED ANSWERS

      // GOOD RESPONSE
      // prompt: `Create an investment portfolio based on the age: ${restaurantTheme} of the user.`,
      
      // ONLY GIVES GENERAL IDEA, EXPLAINS CONCEPTS NO VALUES
      prompt: `Analyse the stock of the comapny ${restaurantTheme} for the user.`,
    });
    return text;
  }
);

const MenuItemDetailedSchema = z.object({
  dishname:z.string(),
  description: z.string(),
  ingredients: z.array(z.string()),
  allergens: z.array(z.string()),
  recepie: z.array(z.string())
});

export const MenuSuggestionFlowWithDetails = ai.defineFlow(
  {
    name : "MenuSuggestionFlowWithDetails",
    inputSchema: z.string(),
    outputSchema:MenuItemDetailedSchema

  },
  async (restaurantTheme) => {
    const {output} = await ai.generate({
      model: gemini15Flash,
      system: `You are an expert in suggesting menu to users`,
      prompt: 
        `Invent a menu item for a ${restaurantTheme} themed restaurant.` +
        `Give a detailed description of the dish, ingredients used, allergens and detailed recepie.`,
      output: {schema: MenuItemDetailedSchema}
    });
    if(output == null){
      throw new Error("Output does not satify the schema.")
    }
    return output;
  }
);

const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'gets the current weather in a given location',
    inputSchema: z.object({
      location: z.string().describe('the location to get the current weather for')
    }),
  },
  async (intput) => {
    // call an api or query a databse
    return{
      temperature: '32C',
      weather: 'sunny',
    };
  }
);

const getCurrentLocation = ai.defineTool(
  {
    name: 'getCurrentLocation',
    description: 'Gets the current location using the browser API',
    outputSchema: z.object({
      latitude: z.number(),
      longitude: z.number(),
      city: z.string(),
      country: z.string(),
    }),
  },
  async (input) => {
    // call the browser api to get the location
    return {
      latitude: 12.9716,
      longitude: 77.5946,
      city: 'Bengaluru',
      country: 'India',
    };
  }
);

export const MenuSuggestionFlowWithDetailsAndLocation = ai.defineFlow(
  {
    name : "MenuSuggestionFlowWithDetailsAndLocation",
    inputSchema: z.string(),
    outputSchema:MenuItemDetailedSchema

  },
  async (restaurantTheme) => {
    const {output} = await ai.generate({
      model: gemini15Flash,
      system: `You are an expert in suggesting menu to users`,
      prompt: 
        `Invent a menu item for a ${restaurantTheme} themed restaurant.` +
        `Give a detailed description of the dish, ingredients used, allergens and detailed recepie.`
        + 'Take into account current weather and the location when suggesting a dish.',
      output: {schema: MenuItemDetailedSchema},
      tools: [getWeather,getCurrentLocation],
    });
    if(output == null){
      throw new Error("Output does not satify the schema.")
    }
    return output;
  }
);

const MenuItemPrepSchema = ai.defineSchema(
  'MenuItemPrepSchema',
  z.object({
    description: z.string(),
    recipe: z.array(z.string()),
    ingredients: z.array(z.string())
  })
);

const MenuItemNutritionSchema = ai.defineSchema(
  'MenuItemNutritionSchema',
  z.object({
    calories: z.number(),
    protein: z.number(),
    carbs: z.number(),
    fats: z.number()
  })
);

export const getCaloriesAndMacroDetails = ai.definePrompt(
  {
    name: 'getCaloriesAndMacroDetails',
    description: 'Gets the calories and macro details of the dish',
    model: gemini20Flash,
    input:{
      schema: MenuItemPrepSchema
    },
    output:{
      schema: MenuItemNutritionSchema
    },
    system: 'You are a nutrition expert working at the national institute of nutritiona and health',
    prompt:
    'Given the dish description, recipe, and ingredients, analyze the provided information and tell the calories and the macro breakdown per 100 grams'
    + 'Description: {{description}}\n'+
    'Recipe: {{recipe}}\n'
    + 'Ingrediendts: {{ingredients}}'
  }
);

// export const getCaloriesAndMacroDetails = ai.prompt('getCaloriesAndMacros')

// const MenuItemWithMacrosSchema = z.object({
//   ...MenuItemDetailedSchema.shape,
//   ...MenuItemNutritionSchema.shape
// });

// export const menuSuggestionWithMacrosFlow = ai.defineFlow(
//   {
//     name: 'menuSuggestionWithMacrosFlow',
//     inputSchema: z.string(),
//     outputSchema: MenuItemWithMacrosSchema,
//   },
//   async (restaurantTheme) => {
//     const menuItem = await MenuSuggestionFlowWithDetailsAndLocation(restaurantTheme);
//     const {output} = await getCaloriesAndMacroDetails({
//       description: menuItem.description,
//       recipe: menuItem.recepie,
//       ingredients: menuItem.ingredients
//     });
//     return {
//       ...menuItem,
//       calories: output['calories'],
//       protein: output['protein'],
//       carbs: output['carbs'],
//       fats: output['fats']
//     };
//   }
// );

// startFlowServer({
//   flows: [MenuSuggestionFlow,MenuSuggestionFlowWithDetailsAndLocation, menuSuggestionWithMacrosFlow,MenuSuggestionFlowWithDetails],
//   port: 3000,
// });