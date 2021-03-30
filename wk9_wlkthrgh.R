### 1. Prepare ###
## 1c. Setup ##

library(tidyverse)
library(tidytext)
library(SnowballC)
library(topicmodels)
library(stm)
library(ldatuning)
library(knitr)
library(LDAvis)

## 2a. Import Forum Data ##
ts_forum_data <- read_csv("data/ts_forum_data.csv", 
                          col_types = cols(course_id = col_character(),
                                           forum_id = col_character(), 
                                           discussion_id = col_character(), 
                                           post_id = col_character()
                          )
)

################################################################################
# ✅ Comprehension Check #   
# Try importing directly from the ECI 588 Github repository. The data for Unit 3 is located in this folder: https://github.com/sbkellogg/eci-588/tree/main/unit-3/data
# Hint: Check the examples from the ?read_csv help file.

# ts_forum_data_github <- read_csv("https://github.com/sbkellogg/eci-588/blob/main/unit-3/data/ts_forum_data.csv")
# i think the above is how you're supposed to import directly from github i got an error message? "warning: 83 parsing failures"
################################################################################

## 2b. Tidy Text for Topic Modeling ##

forums_tidy <- ts_forum_data %>%
  unnest_tokens(output = word, input = post_content) %>%
  anti_join(stop_words, by = "word")

forums_tidy %>%
  count(word, sort = TRUE)

# forums_tidy

################################################################################
# ✅ Comprehension Check #   
# Use the filter() and grepl() functions introduced in Unit 1. Section 3b to filter for rows in our ts_forum_data data frame that contain the terms “agree” and “time” and another term or terms of your choosing. Select a random sample of 10 posts using the sample_n() function for your terms and answer the following questions:
#   
forum_quotes <- ts_forum_data %>%
  select(post_content) %>% 
  filter(grepl('agree', post_content)) %>% 
  filter(grepl('time', post_content)) %>%
  filter(grepl('students', post_content))

sample_quotes <- sample_n(forum_quotes, 10)

#   What, if anything, do these posts have in common?
#   
# I'm seeing a lot of comments related to inspiration instructors gleaned about how to better serve their students based on what they picked up in the course/ from their converstions. 

#   What topics or themes might be apparent, or do you anticipate emerging, from our topic modeling?
#
# strategies for improving teaching



# ################################################################################

# Creating a document term matrix 
forums_dtm <- forums_tidy %>%
  count(post_id, word) %>%
  cast_dtm(post_id, word, n)

################################################################################
# ✅ Comprehension Check
# Take a look at our forums_dtm object in the console and answer the following question:
#   
#   What “class” of object is forums_dtm?
# class(forums_dtm)
# [1] "DocumentTermMatrix"    "simple_triplet_matrix"

#   How many unique documents and terms are included our matrix?
# for me it's listing 5766 documents, 13620 terms...which is slightly different than what is in the walkthorugh, but so was my original #of obs. as compared to the wwalkthrough (forums_tidy = 192,159 vs. 165,720 ??)

#   Why might there be fewer documents/posts than were in our original data frame?
# some have been filtered out/discarded in our processing steps

#   What exactly is meant by “sparsity”?
# sparsity refers to the cells that have zero counts. so in our case 100% sparsity means there are no elements that are empty.
################################################################################

## 2c. To stem or not to stem? ##

temp <- textProcessor(ts_forum_data$post_content, 
                      metadata = ts_forum_data,  
                      lowercase=TRUE, 
                      removestopwords=TRUE, 
                      removenumbers=TRUE,  
                      removepunctuation=TRUE, 
                      wordLengths=c(3,Inf),
                      stem=TRUE,
                      onlycharacter= FALSE, 
                      striphtml=TRUE, 
                      customstopwords=NULL)

meta <- temp$meta
vocab <- temp$vocab
docs <- temp$documents

stemmed_forums <- ts_forum_data %>%
  unnest_tokens(output = word, input = post_content) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(stem = wordStem(word))

stemmed_forums

################################################################################
# ✅ Comprehension Check
# Complete the following code using what we learned in the section on Creating a Document Term Matrix and answer the following questions:
#   
stemmed_dtm <- ts_forum_data %>%
  unnest_tokens(output = word, input = post_content) %>%
  anti_join(stop_words, by = "word") %>%
  mutate(stem = wordStem(word)) %>%
  count(post_id, stem, sort = TRUE) %>%
  cast_dtm(post_id, stem, n)

stemmed_dtm


#   How many fewer terms are in our stemmed document term matrix?
# approximately 36,000

#   Did stemming words significantly reduce the sparsity of the network?
# sparsity remains at 100%

#   Hint: Make sure your code includes stem counts rather than word counts.

################################################################################

### 3. Model ###

## 3a. Fitting a Topic Modeling w/ LDA ##

n_distinct(ts_forum_data$forum_name)

forums_lda <- LDA(forums_dtm, 
                  k = 21, 
                  control = list(seed = 588)
)

forums_lda

## 3b. Fitting a Structural TOpic model ##

docs <- temp$documents 
meta <- temp$meta 
vocab <- temp$vocab 

forums_stm <- stm(documents=docs, 
                  data=meta,
                  vocab=vocab, 
                  prevalence =~ course_id + forum_id,
                  K=21,
                  max.em.its=25,
                  verbose = FALSE)

forums_stm

plot.STM(forums_stm, n = 5)
plot(forums_stm, n = 5)

################################################################################
# ✅ Comprehension Check
# Fit a model for both LDA and STM using different values for K and answer the following questions:
#   
# fitting a model for 10 topics instead of 21

forums_lda_10 <- LDA(forums_dtm, 
                  k = 10, 
                  control = list(seed = 588)
)

forums_lda_10

forums_stm_10 <- stm(documents=docs, 
                  data=meta,
                  vocab=vocab, 
                  prevalence =~ course_id + forum_id,
                  K=10,
                  max.em.its=25,
                  verbose = FALSE)

forums_stm_10

plot.STM(forums_stm_10, n = 5)


#   What topics appear to be similar to those using 20 topics for K?
# topic 
#specifying less topics definitely makes the results less nuanced and harder to pinpoint what exactly the 'topic' is...both values for k seem to turn up themes of assisting students, teaching stats, and student engagement. 

#   Knowing that you don’t have as much context as I do, how might you interpret one of these latent topics or themes using the key terms assigned?
#topic 3 for me reads: 'statist, can, learn, engage, start', -- seems to have something to do with motivating students.

#   What topic emerged that seem dramatically different and how might you interpret this topic?
# my analysis is differing from what i'm seeing in the walkthrough, but in the walkthorugh I noteced that topic 6 was 'coaster, roller, steel, speed, present' -- which i would guess has something to do with a specific math problem or example that was discussed?

################################################################################

## 3c. Finding K ##

k_metrics <- FindTopicsNumber(
  forums_dtm,
  topics = seq(10, 75, by = 5),
  metrics = "Griffiths2004",
  method = "Gibbs",
  control = list(),
  mc.cores = NA,
  return_models = FALSE,
  verbose = FALSE,
  libpath = NULL
)

FindTopicsNumber_plot(k_metrics)

# could run the below but don't need to for the walthrough because it will take forever
# findingk <- searchK(docs, 
#                     vocab, 
#                     K = c(5:15),
#                     data = meta, 
#                     verbose=FALSE)
# 
# plot(findingk)

# LDAvis explorere
toLDAvis(mod = forums_stm, docs = docs)

### 4a. Exploring Beta Values ###

terms(forums_lda, 5)

tidy_lda <- tidy(forums_lda)

tidy_lda

top_terms <- tidy_lda %>%
  group_by(topic) %>%
  slice_max(beta, n = 5, with_ties = FALSE) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  group_by(topic, term) %>%    
  arrange(desc(beta)) %>%  
  ungroup() %>%
  ggplot(aes(beta, term, fill = as.factor(topic))) +
  geom_col(show.legend = FALSE) +
  scale_y_reordered() +
  labs(title = "Top 5 terms in each LDA topic",
       x = expression(beta), y = NULL) +
  facet_wrap(~ topic, ncol = 4, scales = "free")

# 4b. Exploring Gamma Values ##

td_beta <- tidy(forums_lda)

td_gamma <- tidy(forums_lda, matrix = "gamma")

td_beta

td_gamma

top_terms <- td_beta %>%
  arrange(beta) %>%
  group_by(topic) %>%
  top_n(7, beta) %>%
  arrange(-beta) %>%
  select(topic, term) %>%
  summarise(terms = list(term)) %>%
  mutate(terms = map(terms, paste, collapse = ", ")) %>% 
  unnest()

gamma_terms <- td_gamma %>%
  group_by(topic) %>%
  summarise(gamma = mean(gamma)) %>%
  arrange(desc(gamma)) %>%
  left_join(top_terms, by = "topic") %>%
  mutate(topic = paste0("Topic ", topic),
         topic = reorder(topic, gamma))

gamma_terms %>%
  select(topic, gamma, terms) %>%
  kable(digits = 3, 
        col.names = c("Topic", "Expected topic proportion", "Top 7 terms"))

plot(forums_stm, n = 7)


# 4c. Reading the Tea Leaves # 

ts_forum_data_reduced <-ts_forum_data$post_content[-temp$docs.removed]

findThoughts(forums_stm,
             texts = ts_forum_data_reduced,
             topics = 2, 
             n = 10,
             thresh = 0.5)

findThoughts(forums_stm,
             texts = ts_forum_data_reduced,
             topics = 16, 
             n = 10,
             thresh = 0.5)

findThoughts(forums_stm,
             texts = ts_forum_data_reduced,
             topics = 3, 
             n = 10,
             thresh = 0.5)

################################################################################
# ✅ Comprehension Check
# Using the STM model you fit from the Section 3 [Comprehension Check] with a different value for K, use the approaches demonstrated in Section 4 to explore and interpret your topics and terms and revisit the following question:
#   
#   Now that you have a little more context, how might you revise your initial interpretation of some of the latent topics or latent themes from your model?

findThoughts(forums_stm_10,
             texts = ts_forum_data_reduced,
             topics = 3, 
             n = 10,
             thresh = 0.5)

# My initial interpretation of topic 3 which for me reads: 'statist, can, learn, engage, start', was that this seems to have something to do with motivating students.
# in looking at some somple responses for this topic it seems to be drawing from comments that were marketing a graduate certificate program?


################################################################################
