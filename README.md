# DBI-RecSys
Movie recommender system for Huawei Video, made for Digital Business Innovation Lab project @Polimi 2018/19

## Goals

### front-end

  - Home with top-pop movie slides
  - Home with recommended contents
  - movie page with youtube trailer
  - Search functionality (helpful to build test profiles)

### back-end

  - REST web-server
  - API to match front-end:
    - `\movies\<movie_id>` + log request by `user_id`
    - `\movies\top_pop`
    - `\movies\rec?user_id=<id>`
    - `\movies?search=<keywords>` + log request by `user_id`
    - `POST \interactions` form `user_id, movie_id, timestamp`
  - Plug-in recommender function, in order to change it after trainings

### recsys

Until we don't get data, I would focus on reproducing some baselines (even part of the CBF vs. genomic CBF).
Normal WF:

  - Import data
  - Split in test & train
  - Tune hyperparameters with CV folds
  - Test baselines on test set.

The winner algorithm can be deployed on the backend.
