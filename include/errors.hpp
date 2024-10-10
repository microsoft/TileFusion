// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <exception>
#include <string>

namespace tilefusion::errors {

class NotImplementedException : public std::exception {
  public:
    NotImplementedException(const char* error = "Not yet implemented!") {
        errorMessage = error;
    }

    // Provided for compatibility with std::exception.
    const char* what() const noexcept { return errorMessage.c_str(); }

  private:
    std::string errorMessage;
};

}  // namespace tilefusion::errors
